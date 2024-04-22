import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from typing import Iterable, List, Tuple
from copy import deepcopy


class PCGrad():
	"""Gradient Surgery for Multi-Task Learning. Projecting Conflicting Gradients

	:param optimizer: OPTIMIZER: optimizer instance.
	:param reduction: str. reduction method.
	"""
	def __init__(self, optimizer, reduction='mean'):
		self.optimizer = optimizer
		self.reduction = reduction
		return

	@torch.no_grad()
	def reset(self):
		self.zero_grad()

	def zero_grad(self):
		return self.optimizer.zero_grad(set_to_none=True)

	def step(self):
		return self.optimizer.step()

	def set_grad(self, grads: List[torch.Tensor]):
		idx: int = 0
		for group in self.optimizer.param_groups:
			for p in group['params']:
				p.grad = grads[idx]
				idx += 1

	def retrieve_grad(self):
		"""Get the gradient of the parameters of the network with specific objective."""

		grad, shape, has_grad = [], [], []
		for group in self.optimizer.param_groups:
			for p in group['params']:
				if p.grad is None:
					shape.append(p.shape)
					grad.append(torch.zeros_like(p, device=p.device))
					has_grad.append(torch.zeros_like(p, device=p.device))
					continue
				shape.append(p.grad.shape)
				grad.append(p.grad.clone())
				has_grad.append(torch.ones_like(p, device=p.device))
		return grad, shape, has_grad

	def pack_grad(self, objectives):
		"""Pack the gradient of the parameters of the network for each objective.

		:param objectives: Iterable[nn.Module]. a list of objectives.
		:return: torch.Tensor. packed gradients.
		"""

		grads, shapes, has_grads = [], [], []
		for objective in objectives:
			self.optimizer.zero_grad(set_to_none=True)
			# objective.backward()
			objective.backward(retain_graph=True)

			grad, shape, has_grad = self.retrieve_grad()

			grads.append(self.flatten_grad(grad))
			has_grads.append(self.flatten_grad(has_grad))
			shapes.append(shape)

		return grads, shapes, has_grads

	def project_conflicting(self, grads: List[torch.Tensor], has_grads: List[torch.Tensor]) -> torch.Tensor:
		"""Project conflicting.
		In our setting, the first task is the main task and th second task is the auxiliary task.

		:param grads: a list of the gradient of the parameters.
		:param has_grads: a list of mask represent whether the parameter has gradient.
		:return: torch.Tensor. merged gradients.
		"""
		shared: torch.Tensor = (torch.stack(has_grads)[0, :]*torch.stack(has_grads)[1, :]).bool()
		pc_grad: List[torch.Tensor] = deepcopy(grads)
		# pc_grad[0]: g_main, pc_grad[1]: g_auxiliary
		g_dot = torch.dot(pc_grad[1], pc_grad[0])
		conflict_flag = False
		if g_dot < 0:
			conflict_flag = True
			pc_grad[1] -= g_dot * pc_grad[0]/(pc_grad[0].norm()**2)
		
		merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
		if self.reduction == 'mean':
			merged_grad[shared] = torch.stack([g[shared] for g in pc_grad]).mean(dim=0)
		elif self.reduction == 'sum':
			merged_grad[shared] = torch.stack([g[shared] for g in pc_grad]).sum(dim=0)
		else:
			exit('invalid reduction method')

		merged_grad[~shared] = torch.stack([g[~shared] for g in pc_grad]).sum(dim=0)
		return merged_grad

	def pc_backward(self, objectives):
		"""Calculate the gradient of the parameters.

		:param objectives: Iterable[nn.Module]. a list of objectives.
		"""

		grads, shapes, has_grads = self.pack_grad(objectives)
		pc_grad = self.project_conflicting(grads, has_grads)
		# print(grads)
		# print(pc_grad)
		pc_grad = self.un_flatten_grad(pc_grad, shapes[0])
		self.set_grad(pc_grad)

	def un_flatten_grad(self, grads: torch.Tensor, shapes: List[int]) -> List[torch.Tensor]:
		"""Unflatten the gradient"""
		idx: int = 0
		un_flatten_grad: List[torch.Tensor] = []
		for shape in shapes:
			length = np.prod(shape)
			un_flatten_grad.append(grads[idx:idx + length].view(shape).clone())
			idx += length
		return un_flatten_grad

	def flatten_grad(self, grads: List[torch.Tensor]) -> torch.Tensor:
		"""Flatten the gradient."""
		return torch.cat([grad.flatten() for grad in grads])
