#ifndef _HEAD_H_
#define _HEAD_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

using namespace amrex;

void GPotential();

void advance(	MultiFab& phi_old, 
		MultiFab& phi_new,
		Array<amrex::MultiFab, AMREX_SPACEDIM>& deriv, 
		MultiFab& mu_old, 
		MultiFab& mu_new,
		Array<amrex::MultiFab, AMREX_SPACEDIM>& mu_der,
		MultiFab& term1,
		MultiFab& term2,
		MultiFab& term3,
		MultiFab& ac,
		MultiFab& eeta,
		Real gamma,
		Real dab,
		Real tau,
		Real dt,
		Real eps,
		Vector<Real>& A,
		Real& B,
		Real& D,
		Vector<Real>& diff,
		Geometry const& geom);
		
void init_phi (	MultiFab& phi_new,
		Real seed,
		Real ncell);	

void init_mu (MultiFab& mu_new, Real A_liq, Real ceq_liq);	 

//void der(amrex::MultiFab& phiold, Array<amrex::MultiFab, AMREX_SPACEDIM>& derivative, amrex::Geometry const& geom);
void dmudt(amrex::MultiFab& mu_new, amrex::MultiFab& mu_old, Array<amrex::MultiFab, AMREX_SPACEDIM>& mu_der, amrex::MultiFab& phi_new, amrex::MultiFab& phi_old, Vector<Real> diffusivity, Vector<Real> A, amrex::Real B, amrex::Real dt, amrex::Geometry const& geom);

void computeterm2(MultiFab& term2, MultiFab& phi_old, Real gamma);
void computeterm3(amrex::MultiFab& mu_old, amrex::MultiFab& term3, amrex::MultiFab& phi_old, Vector<amrex::Real> A, amrex::Real B, amrex::Real D);
		
// void calc_eeta(Array<MultiFab, AMREX_SPACEDIM>& derivative, MultiFab& eeta, Real delta);
// void anisotropy (MultiFab& eeta, MultiFab& term1, Array<amrex::MultiFab,AMREX_SPACEDIM>& derivative, Real delta, Real gamma, Geometry const& geom);

void calc_h(MultiFab& phi_new, MultiFab& h);

#endif

