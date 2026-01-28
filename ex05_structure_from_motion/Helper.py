# ==============================================================================
# File        : Helper.py
# Author      : Elham Khazaee
# Course      : Photogrammetric Computer Vision
# Program     : M.Sc. Geodesy and Geoinformation Science, TU Berlin
# Description : Helper utilities for Exercise 05 –
# Structure from Motion (SfM), including geometric transformations,
# optimization support, and visualization utilities.
#============================================================================

import numpy as np


def hom2eucl(point):
    point = np.asarray(point, dtype=np.float32)
    return point[:-1] / point[-1]


def translationMatrix(x, y, z):
    result = np.eye(4, dtype=np.float32)
    result[0, 3] = x
    result[1, 3] = y
    result[2, 3] = z
    return result


def rotationMatrixX(radAngle):
    result = np.eye(4, dtype=np.float32)
    c = np.cos(radAngle)
    s = np.sin(radAngle)
    result[1, 1] = c
    result[1, 2] = -s
    result[2, 1] = s
    result[2, 2] = c
    return result


def rotationMatrixY(radAngle):
    result = np.eye(4, dtype=np.float32)
    c = np.cos(radAngle)
    s = np.sin(radAngle)
    result[0, 0] = c
    result[0, 2] = s
    result[2, 0] = -s
    result[2, 2] = c
    return result


def rotationMatrixZ(radAngle):
    result = np.eye(4, dtype=np.float32)
    c = np.cos(radAngle)
    s = np.sin(radAngle)
    result[0, 0] = c
    result[0, 1] = -s
    result[1, 0] = s
    result[1, 1] = c
    return result


class OptimizationProblem:
    class JacobiMatrix:
        def multiply(self, dst, src):
            raise NotImplementedError

        def transposedMultiply(self, dst, src):
            raise NotImplementedError

        def computeDiagJtJ(self, dst):
            raise NotImplementedError

    class State:
        def clone(self):
            raise NotImplementedError

        def computeResiduals(self, residuals):
            raise NotImplementedError

        def computeJacobiMatrix(self, dst):
            raise NotImplementedError

        def update(self, update, dst):
            raise NotImplementedError

    def __init__(self):
        self.m_numUpdateParameters = 0
        self.m_numResiduals = 0

    def getNumUpdateParameters(self):
        return self.m_numUpdateParameters

    def getNumResiduals(self):
        return self.m_numResiduals

    def createJacobiMatrix(self):
        raise NotImplementedError


class LevenbergMarquardt:
    def __init__(self, optimizationProblem, initialState):
        self.m_optimizationProblem = optimizationProblem
        self.m_state = initialState
        self.m_newState = self.m_state.clone()
        self.m_jacobiMatrix = self.m_optimizationProblem.createJacobiMatrix()

        num_res = self.m_optimizationProblem.getNumResiduals()
        num_params = self.m_optimizationProblem.getNumUpdateParameters()

        self.m_residuals = np.zeros((num_res,), dtype=np.float32)
        self.m_newResiduals = np.zeros((num_res,), dtype=np.float32)
        self.m_update = np.zeros((num_params,), dtype=np.float32)
        self.m_JtR = np.zeros((num_params,), dtype=np.float32)
        self.m_diagonal = np.zeros((num_params,), dtype=np.float32)

        self.m_state.computeResiduals(self.m_residuals)
        self.m_lastError = float(np.dot(self.m_residuals, self.m_residuals))
        self.m_damping = 1.0

    def iterate(self):
        self.m_state.computeJacobiMatrix(self.m_jacobiMatrix)
        self.m_jacobiMatrix.transposedMultiply(self.m_JtR, self.m_residuals)

        self.m_jacobiMatrix.computeDiagJtJ(self.m_diagonal)
        self.m_diagonal *= self.m_damping

        tmp = np.zeros((self.m_optimizationProblem.getNumResiduals(),), dtype=np.float32)

        def matMul(dst, src):
            dst[:] = 0.0
            self.m_jacobiMatrix.multiply(tmp, src)
            self.m_jacobiMatrix.transposedMultiply(dst, tmp)
            dst[:] = dst + self.m_diagonal * src

        self.m_update.fill(0.0)
        r = self.m_JtR.copy()
        p = r.copy()
        residual = float(np.dot(r, r))
        if residual > 1e-8:
            for _ in range(100):
                Ap = np.zeros_like(self.m_update)
                matMul(Ap, p)
                pAp = float(np.dot(p, Ap))
                if abs(pAp) < 1e-12:
                    break
                alpha = residual / pAp
                self.m_update += alpha * p
                r -= alpha * Ap

                newResidual = float(np.dot(r, r))
                if newResidual < 1e-8:
                    break
                p = p * (newResidual / residual) + r
                residual = newResidual

        self.m_state.update(self.m_update, self.m_newState)
        self.m_newState.computeResiduals(self.m_newResiduals)
        newError = float(np.dot(self.m_newResiduals, self.m_newResiduals))

        if newError < self.m_lastError:
            self.m_lastError = newError
            self.m_state, self.m_newState = self.m_newState, self.m_state
            self.m_residuals, self.m_newResiduals = self.m_newResiduals, self.m_residuals
            self.m_damping *= 0.9
        else:
            self.m_damping *= 2.0

    def getLastError(self):
        return self.m_lastError

    def getDamping(self):
        return self.m_damping

    def getState(self):
        return self.m_state
