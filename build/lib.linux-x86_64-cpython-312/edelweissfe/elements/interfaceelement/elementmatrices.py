#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _         _____ _____
# | ____|__| | ___| |_      _____(_)___ ___|  ___| ____|
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __| |_  |  _|
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \  _| | |___
# |_____\__,_|\___|_| \_/\_/ \___|_|___/___/_|   |_____|
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#  2017 - today
#
#  Alexandros Stathas alexandros.stathas@boku.ac.at
#  Matthias Neuner matthias.neuner@uibk.ac.at
#
#  This file is part of EdelweissFE.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissFE.
#  ---------------------------------------------------------------------

from basix import CellType, ElementFamily
import numpy as np
from numpy._core.fromnumeric import shape


def computeJacobian(x: np.ndarray, element, quad_points, dim: int):
    """Get the Jacobi matrix for the element calculation.

    Parameters
    ----------
    x
        Global coordinates of the element points.
    element
        Element object.
    quad_points
        Quadrature points.
    quad_weights
        Quadrature weights.
    nInt
        Number of integration points.
    nNodes
        Number of nodes the element has.
    dim
        Dimension the element has.
    """
    # Define interface element and quadrature rule
    if (dim-1) == 1:

        # Initialize Jacobian matrices at each quadrature point
        jacobians = np.zeros((len(quad_points), dim, 1))
        # number of quadrature points, scalar Jacobian matrix

        # Compute gradient matrices (shape function derivatives at quadrature points)
        gradients = element.tabulate(1, quad_points)[1:]
         
        for qp in range(len(quad_points)):  # Loop over quadrature points
            dphi_dxi = gradients[0,qp, :, :]  # dφ/dξ for all nodes
            
            # Compute Jacobian matrix
            J = np.zeros((2, 1))
            J[0, 0] = np.einsum('i,ip->p',x[:, 0] , dphi_dxi)  # dx/dξ
            J[1, 0] = np.einsum('i,ip->p',x[:, 1] , dphi_dxi)  # dy/dξ

            jacobians[qp] = J  # Store Jacobian for this quadrature point
            #print('jacobians',jacobians)

    elif (dim-1) == 2:
        # Initialize Jacobian matrices at each quadrature point
        jacobians = np.zeros((len(quad_points), dim, 2))
        # number of quadrature points, 3x2 Jacobian matrix

        # Compute gradient matrices
        # (shape function derivatives at quadrature points)

        gradients = element.tabulate(1, quad_points)[1:] # we only keep one component because the element has the same dependence in xi,eta

        for qp in range(len(quad_points)):  # Loop over quadrature points
            dphi_dxi = gradients[0, qp, :, :]  # dφ/dξ for all nodes
            dphi_deta = gradients[1, qp, :, :]  # dφ/dη for all nodes
            
            # Compute Jacobian matrix
            J = np.zeros((3, 2))

            J[0, 0] = np.einsum('i,ip->p', x[:, 0] , dphi_dxi)  # dx/dξ
            J[0, 1] = np.einsum('i,ip->p', x[:, 0] , dphi_deta)  # dx/dη
            J[1, 0] = np.einsum('i,ip->p', x[:, 1] , dphi_dxi)  # dy/dξ
            J[1, 1] = np.einsum('i,ip->p', x[:, 1] , dphi_deta)  # dy/dη
            J[2, 0] = np.einsum('i,ip->p', x[:, 2] , dphi_dxi)
            J[2, 1] = np.einsum('i,ip->p', x[:, 2] , dphi_deta)

            jacobians[qp] = J  # Store Jacobian for this quadrature point
    #print('QUAD POINTS:\n', quad_points)
    #print('shape functions', element.tabulate(0, quad_points))
    #print('gradients pure\n', element.tabulate(1, quad_points)[1:].shape)
    #print('gradients.shape:', gradients.shape)
    #print('reference gradients scalar case:\n',gradients)
    #print('jacobians.shape:\n', jacobians.shape)
    #print('jacobians', jacobians)
    return jacobians, gradients


def compute_grad(x: np.ndarray, element, quad_points, nInt: int, dim: int):
    """
    Get the gradient operator for each component of the vector finite element.

    Parameters
    ----------
    x
        Global coordinates of the element points.
    nInt
        Number of integration points.
    nNodes
        Number of nodes the element has.
    dim
        Dimension the element has.

    The B operator for the interface elements uses the full gradient
    for the shear strain not the symmetric part only
    """

    J_at_GPs, gradients = computeJacobian(x, element, quad_points, dim)

    grad_phi_x_y_at_GPs_shape = np.append(J_at_GPs.shape[:2],gradients.shape[2:])

    grad_phi_x_y_at_GPs = np.zeros(grad_phi_x_y_at_GPs_shape)  # Shape (GP,node,dim)

    # The elements ar embedded in a higher dimensional space 
    # We follow the metric tensor approach to actually compute the gradients from refernce ti global configuration
    G = np.zeros((J_at_GPs.shape[0],J_at_GPs.shape[2],J_at_GPs.shape[2]))

    sqrt_detG = np.zeros((J_at_GPs.shape[0]))
    for qp in range(nInt):
        G[qp] = np.einsum('ij,jk->ik',J_at_GPs[qp].T,J_at_GPs[qp])
        sqrt_detG[qp] = np.sqrt(np.linalg.det(G[qp]))

        G_inv = np.linalg.inv(G[qp])

        J_pseudo_inverse_transpose = np.einsum('ij,jk->ik', J_at_GPs[qp], G_inv) 

        if dim == 2:

            grad_phi_x_y_at_GPs[qp] = np.einsum(
                "ij,jad->iad",J_pseudo_inverse_transpose, gradients[:,qp,:,:])    
        elif dim == 3:

            grad_phi_x_y_at_GPs[qp] = np.einsum('ij,jad->iad', J_pseudo_inverse_transpose, gradients[:,qp,:,:] )
                
    grad_phi_x_y_at_GPs_stacked = np.repeat(
        grad_phi_x_y_at_GPs[np.newaxis, :, :, :], dim, axis=0
    )
    
    return grad_phi_x_y_at_GPs_stacked, sqrt_detG


def compute_surface_grad(
    x: np.ndarray, element, quad_points, nInt: int, dim: int
):
    """
    Get the surface gradient operator for each component of the vector
    finite element.

    Parameters
    ----------
    x
        Global coordinates of the element points.
    nInt
        Number of integration points.
    nNodes
        Number of nodes the element has.
    dim
        Dimension the element has.

    """

    _, _, T = interface_geometry(x, element, quad_points, nInt, dim)
    # T is of Shape: (component, dim, GP )
    grad_phi_x_y_at_GPs_stacked, _ = compute_grad(x, element, quad_points, nInt, dim)
    # Shape: (node, component, dim, GP)
    if dim == 2:
        surface_grad_phi_x_y_at_GPs_stacked = np.einsum(
            "iqja, jkq -> aikq", grad_phi_x_y_at_GPs_stacked[:,:,:,:,0], T
        )
    elif dim == 3:
        surface_grad_phi_x_y_at_GPs_stacked = np.einsum(
            "iqja, jkq -> aikq", grad_phi_x_y_at_GPs_stacked[:,:,:,:,0], T
        )
        
    print('norm of difference surface_grad to global grad:\n',\
          np.linalg.norm(surface_grad_phi_x_y_at_GPs_stacked\
          -(grad_phi_x_y_at_GPs_stacked[:,:,:,:,0].transpose(3,0,2,1))))
    return surface_grad_phi_x_y_at_GPs_stacked


def compute_surface_div_grad(
    x: np.ndarray, element, quad_points, nInt: int, dim: int
):
    """
    Get the divergence of the surface the gradient operator for each
    component of the vector finite element.
    This operator is not necessary for the
    solution of the variational problem. It is used for evaluating the jump
    of the stress vector at the interface.
    Thus it is important for post processing and converegence analyses

    To make use of this operator we need to evaluate the normal vector
    at the nodes of each element.
    We need the same for the material but we can assume that
    the local average is enough

    If more detail is needed an enhanced FE method
    (e.g. discontinuous galerkin) should be used...

    Parameters
    ----------
    x
        Global coordinates of the element points.
    nInt
        Number of integration points.
    nNodes
        Number of nodes the element has.
    dim
        Dimension the element has.

    """

    _, _, T = interface_geometry(x, element, quad_points, nInt, dim)
    # T is of Shape: (component, dim, GP )
    grad_phi_x_y_at_GPs_stacked, _ = compute_grad(x, element, quad_points, nInt, dim)

    # Shape: (node, component, dim, GP)
    if dim == 2:
        surface_div_phi_x_y_at_GPs_stacked = np.einsum(
        "iqjab, ijq -> abq", grad_phi_x_y_at_GPs_stacked, T
        )
    elif dim ==3:
        surface_div_phi_x_y_at_GPs_stacked = np.einsum(
        "iqjab, ijq -> abq", grad_phi_x_y_at_GPs_stacked, T
        )
    # Shape: (node, GP)

    return surface_div_phi_x_y_at_GPs_stacked


def computeNOperator(x: np.ndarray, element, quad_points, dim: int):
    """Get the N operator containing the shape functions.

    Parameters
    ----------
    x
        Gloabal coordinates of the element points
    element
        Element object
    nNodes
        Number of nodes the element has.
    dim
        Dimension the element has.
    """
    # Define interface element and quadrature rule

    N = element.tabulate(0, quad_points)[0]
    #print('N', N.shape) 
    # Get the stacked vector of interpolation functions with
    # Shape: (node, component, GP)
    N_stacked = np.repeat(N[np.newaxis,:, :, :], dim, axis=0)
    # Shape: (component, GP, node)
    
    N_stacked = np.transpose(N_stacked, (2, 0, 1, 3))
    # Shape: (node, component, GP),
    # In order to preserve more compatibility with the implemented functions
    #print('N_stacked',N_stacked.shape)
    return N_stacked


# CALCULATE GEOMETRIC QUANTITIES
def interface_geometry(x: np.ndarray, element, quad_points, nInt: int, dim: int):
    """
    We calculate the normal vector using math:: z = g(x,y)=math::sum Ν^A(x,y)z^A
    Then n = math::left(math::grad g, 1) (+1 shows that the normal point towrds
    increasing z)
    the normal is then stored at each GP as  [g_{,x}, g_{,y}, 1]
    """
    grad_g = compute_grad(x, element, quad_points, nInt, dim)[0][0] # I only need one component

    shape_functions_eval_at_quad_points = element.tabulate(0,quad_points)[0]

    normal = np.zeros((quad_points.shape[0],grad_g.shape[1]))

    for qp in range(grad_g.shape[0]):
        
        if dim == 2:
            # 1D element in 2D: Rotate tangent (-t_y, t_x)
            tangents = np.einsum('ijk,jk->ik',grad_g[qp], x)  # Single tangent vector
            normal[qp] = np.array([-tangents[0,1], tangents[0,0]]) #I take the normal to point upwards 
            
        else:
            # General case (2D element in 3D, 3D element in 4D, etc.)

            tangents = np.einsum('ijk,jk->ik',grad_g[qp, :, :, :], x)  # Extract tangent vectors
            
            normal[qp] = np.cross(tangents[0],tangents[1])  # Cross product of the tangent vectors
        
        if np.linalg.norm(normal[qp])>1e-16: #Avoid the zero vector
            normal[qp] /= np.linalg.norm(normal[qp], keepdims=True) 
    
    N = np.einsum("qi,qj->ijq", normal, normal) #We keep the Gauss point dimension at the end to assure compatibility 
                                                #with the other functions present in the material
    I = np.broadcast_to(np.eye(x.shape[1])[:, :, np.newaxis], N.shape)

    T = I - N

    return normal, N, T

# Assignment routines for the stiffness matrices of the finte element
def assign_K_jumpu_jumpv(grad, Nbasis, H_inv_ij, i):

    K_local_shape = grad.shape
    K_local_dim = K_local_shape[0] * K_local_shape[3]#2
    K_jumpu_jumpv = np.zeros((int(2 * K_local_dim), int(2 * K_local_dim)))
    # We double the matrix beause we have double number of nodes side + side -

    # We perform direct assignment
    for node_A in range(K_local_shape[3]):
        for node_B in range(K_local_shape[3]):
            for component_i in range(K_local_shape[0]):
                for component_j in range(K_local_shape[0]):
                    A = int(node_A * K_local_shape[0]) + component_i
                    B = int(node_B * K_local_shape[0]) + component_j

                    K_jumpu_jumpv[A, B] = (
                        Nbasis[node_A, component_i]
                        * H_inv_ij[component_i, component_j]
                        * Nbasis[node_B, component_j]
                    )

                    K_jumpu_jumpv[A, K_local_dim + B] = (
                        -Nbasis[node_A,component_i]
                        * H_inv_ij[component_i, component_j]
                        * Nbasis[node_B,component_j]
                    )

                    K_jumpu_jumpv[K_local_dim + A, B] = (
                        -Nbasis[node_A, component_i]
                        * H_inv_ij[component_i, component_j]
                        * Nbasis[node_B, component_j]
                    )

                    K_jumpu_jumpv[K_local_dim + A, K_local_dim + B] = (
                        Nbasis[node_A, component_i]
                        * H_inv_ij[component_i, component_j]
                        * Nbasis[node_B, component_j]
                    )

    return K_jumpu_jumpv


def assign_K_grad_s_u_grad_s_v(grad, grad_s, Z_ijkl, i):

    K_local_shape = grad.shape
    K_local_dim = K_local_shape[0] * K_local_shape[3]# for line

    K_grad_s_u_grad_s_v = np.zeros((int(2 * K_local_dim), int(2 * K_local_dim)))
    # We double the matrix beause we have double number of nodes side + side -
    # We perform direct assignment
    for node_A in range(K_local_shape[3]):
        for node_B in range(K_local_shape[3]):
            for component_i in range(K_local_shape[0]):
                for component_k in range(K_local_shape[0]):
                    A = int(node_A * K_local_shape[0]) + component_i
                    B = int(node_B * K_local_shape[0]) + component_k

                    K_grad_s_u_grad_s_v[A, B] = (
                        1/ 4
                        * np.einsum(
                            "j, jl, l->",
                            grad_s[node_A, :, component_i],
                            Z_ijkl[component_i, :, component_k, :],
                            grad_s[node_B, :, component_k],

                        )
                    )

                    K_grad_s_u_grad_s_v[A, K_local_dim + B] = (
                        1
                        / 4
                        * np.einsum(
                            "j, jl, l->",
                            grad_s[node_A, :, component_i],
                            Z_ijkl[component_i, :, component_k, :],
                            grad_s[node_B, :, component_k],
                        )
                    )

                    K_grad_s_u_grad_s_v[K_local_dim + A, B] = (
                        1
                        / 4
                        * np.einsum(
                            "j, jl, l->",
                            grad_s[node_A, :, component_i],
                            Z_ijkl[component_i, :, component_k, :],
                            grad_s[node_B, :, component_k],
                        )
                    )

                    K_grad_s_u_grad_s_v[K_local_dim + A, K_local_dim + B] = (
                        1
                        / 4
                        * np.einsum(
                            "j, jl, l->",
                            grad_s[node_A, :, component_i],
                            Z_ijkl[component_i, :, component_k, :],
                            grad_s[node_B, :, component_k],
                        )
                    )

    return K_grad_s_u_grad_s_v


def assign_K_jump_u_grad_s_v(grad, grad_s, Nbasis, H_inv_nF_ijk, i):

    K_local_shape = grad.shape
    K_local_dim = K_local_shape[0] * K_local_shape[3]
    K_jump_u_grad_s_v = np.zeros((int(2 * K_local_dim), int(2 * K_local_dim)))
    # We double the matrix beause we have double number of nodes side + side -

    # We perform direct assignment
    for node_A in range(K_local_shape[3]):
        for node_B in range(K_local_shape[3]):
            for component_i in range(K_local_shape[0]):
                for component_k in range(K_local_shape[0]):
                    A = int(node_A * K_local_shape[0]) + component_i
                    B = int(node_B * K_local_shape[0]) + component_k

                    K_jump_u_grad_s_v[A, B] = (
                        1/ 2
                        * np.einsum(
                            "j, j->",
                            grad_s[node_A, :, component_i],
                            H_inv_nF_ijk[component_i, :, component_k],
                        )
                        * Nbasis[node_B, component_k]
                    )

                    K_jump_u_grad_s_v[A, K_local_dim + B] = (
                        -1/ 2
                        * np.einsum(
                            "j, j->",
                            grad_s[node_A, :, component_i],
                            H_inv_nF_ijk[component_i, :, component_k],
                        )
                        * Nbasis[node_B, component_k]
                    )

                    K_jump_u_grad_s_v[K_local_dim + A, B] = (
                        1/ 2
                        * np.einsum(
                            "j, j->",
                            grad_s[node_A, :, component_i],
                            H_inv_nF_ijk[component_i, :, component_k],
                        )
                        * Nbasis[node_B, component_k]
                    )

                    K_jump_u_grad_s_v[K_local_dim + A, K_local_dim + B] = (
                        -1/ 2
                        * np.einsum(
                            "j, j->",
                            grad_s[node_A, :, component_i],
                            H_inv_nF_ijk[component_i, :, component_k],
                        )
                        * Nbasis[node_B, component_k]
                    )


    return K_jump_u_grad_s_v

def assign_K_grad_s_u_jump_v(grad, grad_s, Nbasis, H_inv_nF_ijk, i):

    K_local_shape = grad.shape
    K_local_dim = K_local_shape[0] * K_local_shape[3]

    K_grad_s_u_jump_v = np.zeros((int(2 * K_local_dim), int(2 * K_local_dim)))
    # We double the matrix beause we have double number of nodes side + side -

    # We perform direct assignment
    for node_A in range(K_local_shape[3]):
        for node_B in range(K_local_shape[3]):
            for component_i in range(K_local_shape[0]):
                for component_j in range(K_local_shape[0]):
                    A = int(node_A * K_local_shape[0]) + component_i
                    B = int(node_B * K_local_shape[0]) + component_j

                    K_grad_s_u_jump_v[A, B] = (
                        1
                        / 2
                        * Nbasis[node_A, component_i]
                        * np.einsum(
                            "k,k->",
                            H_inv_nF_ijk[component_i, component_j, :],
                            grad_s[node_B, :, component_j],
                        )
                    )

                    K_grad_s_u_jump_v[K_local_dim + A, B] = (
                        1
                        / 2
                        * Nbasis[node_A, component_i]
                        * np.einsum(
                            "k,k->",
                            H_inv_nF_ijk[component_i, component_j, :],
                            grad_s[node_B, :, component_j],
                        )
                    )

                    K_grad_s_u_jump_v[K_local_dim + A, K_local_dim + B] = (
                        -1
                        / 2
                        * Nbasis[node_A, component_i]
                        * np.einsum(
                            "k,k->",
                            H_inv_nF_ijk[component_i, component_j, :],
                            grad_s[node_B, :, component_j],
                        )
                    )
    return K_grad_s_u_jump_v


def assign_P_jumpv(grad, N, force, i):

    P_local_shape = grad.shape
    P_local_dim = P_local_shape[0] * P_local_shape[3]

    P_jumpv = np.zeros((int(2 * P_local_dim), 1))
    # We double the matrix beause we have double number of nodes side + side -

    for node_A in range(P_local_shape[3]):
        for component_i in range(P_local_shape[0]):
            A = int(node_A * P_local_shape[0]) + component_i
            P_jumpv[A] = N[node_A, component_i] * force[component_i]
            P_jumpv[P_local_dim + A] = (
                -N[node_A, component_i] * force[component_i]
            )

    return P_jumpv


def assign_P_grad_s_v(grad, grad_s, surface_stress, i):

    P_local_shape = grad.shape
    P_local_dim = P_local_shape[0] * P_local_shape[3]
    P_grad_s_v = np.zeros((int(2 * P_local_dim), 1))
    # We double the matrix beause we have double number of nodes side + side -

    for node_A in range(P_local_shape[3]):
        for component_i in range(P_local_shape[0]):
            A = int(node_A * P_local_shape[0]) + component_i

            P_grad_s_v[A] = (
                1/ 2
                * np.einsum(
                    "j,j->",
                    grad_s[node_A, :, component_i],
                    surface_stress[component_i, :],
                )
            )

            P_grad_s_v[P_local_dim + A] = (
                1
                / 2
                * np.einsum(
                    "j,j->",
                    grad_s[node_A, :, component_i],
                    surface_stress[component_i, :],
                )
            )

    return P_grad_s_v
