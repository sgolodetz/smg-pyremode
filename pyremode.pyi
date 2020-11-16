from smg.pyopencv import CVMat1b


# CLASSES

class Depthmap:
	def __init__(self, width: int, height: int, fx: float, fy: float, cx: float, cy: float): ...

	def get_converged_count(self) -> int: ...
	def get_converged_percentage(self) -> float: ...
	def get_convergence_map(self) -> CVMat1i: ...
	def get_denoised_depthmap(self, lamb: float = 0.5, iterations: int = 200) -> CVMat1f: ...
	def get_depthmap(self) -> CVMat1f: ...
	def set_reference_image(self, img_curr: CVMat1b, t_curr_world: SE3f, min_depth: float, max_depth: float) -> bool: ...
	def update(self, img_curr: CVMat1b, t_curr_world: SE3f) -> None: ...

	@staticmethod
	def scale_mat(depthmap: CVMat1f) -> CVMat3b: ...

class PinholeCamera:
	fx: float
	fy: float
	cx: float
	cy: float

	def __init__(self, fx: float = 0.0, fy: float = 0.0, cx: float = 0.0, cy: float = 0.0): ...

class SE3f:
	def __init__(self, qw: float, qx: float, qy: float, qz: float, tx: float, ty: float, tz: float): ...
	def data(self, row: int, col: int) -> float: ...
	def inv(self) -> SE3f: ...

# ENUMERATIONS

class EConvergenceState(int):
	pass

UPDATE: EConvergenceState
CONVERGED: EConvergenceState
BORDER: EConvergenceState
DIVERGED: EConvergenceState
NO_MATCH: EConvergenceState
NOT_VISIBLE: EConvergenceState

# import numpy as np
#
#
# # CLASSES
#
# class CVMat1d(np.ndarray):
# 	@staticmethod
# 	def zeros(rows: int, cols: int) -> CVMat1d: ...
#
# class CVMat1f(np.ndarray):
# 	@staticmethod
# 	def zeros(rows: int, cols: int) -> CVMat1f: ...
#
# class CVMat3b(np.ndarray):
# 	@staticmethod
# 	def zeros(rows: int, cols: int) -> CVMat3b: ...
#
# class System:
# 	def __init__(self, voc_file: str, settings_file: str, sensor: ESensor, use_viewer: bool): ...
# 	def track_monocular(self, im: CVMat3b, timestamp: float) -> CVMat1d: ...
# 	def track_rgbd(self, im: CVMat3b, depthmap: CVMat1f, timestamp: float) -> CVMat1d: ...
#
# # ENUMERATIONS
#
# class ESensor(int):
# 	pass
#
# MONOCULAR: ESensor
# STEREO: ESensor
# RGBD: ESensor
