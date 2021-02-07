// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2019 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: keir@google.com (Keir Mierle)
//
// A simple implementation of N-dimensional dual numbers, for automatically
// computing exact derivatives of functions.
//
// While a complete treatment of the mechanics of automatic differentiation is
// beyond the scope of this header (see
// http://en.wikipedia.org/wiki/Automatic_differentiation for details), the
// basic idea is to extend normal arithmetic with an extra element, "e," often
// denoted with the greek symbol epsilon, such that e != 0 but e^2 = 0. Dual
// numbers are extensions of the real numbers analogous to complex numbers:
// whereas complex numbers augment the reals by introducing an imaginary unit i
// such that i^2 = -1, dual numbers introduce an "infinitesimal" unit e such
// that e^2 = 0. Dual numbers have two components: the "real" component and the
// "infinitesimal" component, generally written as x + y*e. Surprisingly, this
// leads to a convenient method for computing exact derivatives without needing
// to manipulate complicated symbolic expressions.
//
// For example, consider the function
//
//   f(x) = x^2 ,
//
// evaluated at 10. Using normal arithmetic, f(10) = 100, and df/dx(10) = 20.
// Next, argument 10 with an infinitesimal to get:
//
//   f(10 + e) = (10 + e)^2
//             = 100 + 2 * 10 * e + e^2
//             = 100 + 20 * e       -+-
//                     --            |
//                     |             +--- This is zero, since e^2 = 0
//                     |
//                     +----------------- This is df/dx!
//
// Note that the derivative of f with respect to x is simply the infinitesimal
// component of the value of f(x + e). So, in order to take the derivative of
// any function, it is only necessary to replace the numeric "object" used in
// the function with one extended with infinitesimals. The class Jet, defined in
// this header, is one such example of this, where substitution is done with
// templates.
//
// To handle derivatives of functions taking multiple arguments, different
// infinitesimals are used, one for each variable to take the derivative of. For
// example, consider a scalar function of two scalar parameters x and y:
//
//   f(x, y) = x^2 + x * y
//
// Following the technique above, to compute the derivatives df/dx and df/dy for
// f(1, 3) involves doing two evaluations of f, the first time replacing x with
// x + e, the second time replacing y with y + e.
//
// For df/dx:
//
//   f(1 + e, y) = (1 + e)^2 + (1 + e) * 3
//               = 1 + 2 * e + 3 + 3 * e
//               = 4 + 5 * e
//
//               --> df/dx = 5
//
// For df/dy:
//
//   f(1, 3 + e) = 1^2 + 1 * (3 + e)
//               = 1 + 3 + e
//               = 4 + e
//
//               --> df/dy = 1
//
// To take the gradient of f with the implementation of dual numbers ("jets") in
// this file, it is necessary to create a single jet type which has components
// for the derivative in x and y, and passing them to a templated version of f:
//
//   template<typename T>
//   T f(const T &x, const T &y) {
//     return x * x + x * y;
//   }
//
//   // The "2" means there should be 2 dual number components.
//   // It computes the partial derivative at x=10, y=20.
//   Jet<double, 2> x(10, 0);  // Pick the 0th dual number for x.
//   Jet<double, 2> y(20, 1);  // Pick the 1st dual number for y.
//   Jet<double, 2> z = f(x, y);
//
//   LOG(INFO) << "df/dx = " << z.v[0]
//             << "df/dy = " << z.v[1];
//
// Most users should not use Jet objects directly; a wrapper around Jet objects,
// which makes computing the derivative, gradient, or jacobian of templated
// functors simple, is in autodiff.h. Even autodiff.h should not be used
// directly; instead autodiff_cost_function.h is typically the file of interest.
//
// For the more mathematically inclined, this file implements first-order
// "jets". A 1st order jet is an element of the ring
//
//   T[N] = T[t_1, ..., t_N] / (t_1, ..., t_N)^2
//
// which essentially means that each jet consists of a "scalar" value 'a' from T
// and a 1st order perturbation vector 'v' of length N:
//
//   x = a + \sum_i v[i] t_i
//
// A shorthand is to write an element as x = a + u, where u is the perturbation.
// Then, the main point about the arithmetic of jets is that the product of
// perturbations is zero:
//
//   (a + u) * (b + v) = ab + av + bu + uv
//                     = ab + (av + bu) + 0
//
// which is what operator* implements below. Addition is simpler:
//
//   (a + u) + (b + v) = (a + b) + (u + v).
//
// The only remaining question is how to evaluate the function of a jet, for
// which we use the chain rule:
//
//   f(a + u) = f(a) + f'(a) u
//
// where f'(a) is the (scalar) derivative of f at a.
//
// By pushing these things through sufficiently and suitably templated
// functions, we can do automatic differentiation. Just be sure to turn on
// function inlining and common-subexpression elimination, or it will be very
// slow!
//
// WARNING: Most Ceres users should not directly include this file or know the
// details of how jets work. Instead the suggested method for automatic
// derivatives is to use autodiff_cost_function.h, which is a wrapper around
// both jets.h and autodiff.h to make taking derivatives of cost functions for
// use in Ceres easier.

use nalgebra::allocator::Allocator;
use nalgebra::storage::Owned;
use nalgebra::{DefaultAllocator, DimName, MatrixMN, Scalar, U1};
// use num_traits::Zero;
use std::fmt::Debug;
use std::ops;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Jet<T: Scalar, N: DimName>
    where DefaultAllocator: Allocator<T, N, U1>,
    Owned<T, N>: Copy,
{
    // The scalar part.
    pub a: T,
    // The infinitesimal part
    // pub v: N,
    pub v: MatrixMN<T, N, U1>,
}

impl<T: Scalar, N: DimName> Jet<T, N>
    where
        DefaultAllocator: Allocator<T, N, U1>,
        Owned<T, N>: Copy,
{
    pub fn new(default_val: T, v_val: T, n: N) -> Jet<T, N>
    {
        Jet {
            a: default_val,
            v: MatrixMN::repeat_generic(n, U1, v_val)
        }
    }
}

impl<N: DimName> ops::Mul for Jet<f64, N>
    where
        DefaultAllocator: Allocator<f64, N, U1>, f64: std::ops::Mul<Output = f64>,
        Owned<f64, N>: Copy,
{
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            a: self.a * rhs.a,
            v: self.a * rhs.v + self.v * rhs.a,
        }
    }
}

// TODO(lucasw) want to have T here instead of hard-coded f64 above, but it isn't working
// because maybe 'Scalar' isn't necessarily multipliable with a Matrix.  If
// the Mul for Matrix could be found it would show what to do, seems like the compiler
// could suggest it automatically.
/*
impl<T: Scalar, N: DimName> ops::Mul for Jet<T, N>
    where
        DefaultAllocator: Allocator<T, N, U1>, T: std::ops::Mul<Output = T>,
        Owned<T, N>: Copy,
{
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            a: self.a * rhs.a,
            v: self.a * rhs.v + self.v * rhs.a,
        }
    }
}
*/

impl<T: Scalar, N: DimName> ops::Add for Jet<T, N>
    where
        DefaultAllocator: Allocator<T, N, U1>, T: std::ops::Add<Output = T>,
        Owned<T, N>: Copy,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            a: self.a + rhs.a,
            v: self.v, // + rhs.v,
        }
    }
}
