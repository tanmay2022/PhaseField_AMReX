#ifndef _CALC_H_
#define _CALC_H_

#include<AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>

#define iph 0
#define imh 1
#define jph 2
#define jmh 3
#define kph 4
#define kmh 5

using namespace amrex;

void init_phi(int i,int j,int k,amrex::Array4<amrex::Real> const& phiNew, amrex::Real seed,Real ncell)
{
	
	if(((i-ncell/2)*(i-ncell/2) + (j-ncell/2)*(j-ncell/2)) < seed){
	
		phiNew(i,j,k) = 1.0;
	
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
void init_mu(int i,int j,int k,amrex::Array4<amrex::Real> const& muNew, amrex::Real A_liq,Real ceq_liq)
{
	
	muNew(i,j,k) = 2*A_liq*ceq_liq;
	
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
//AMREX_GPU_DEVICE AMREX_FORCE_INLINE
/*
void compute_ac(int i,int j,int k, amrex::Array4<amrex::Real const> const& phiOld, amrex::Array4<amrex::Real> const& ac_val, amrex::Real dab, amrex::Geometry const& geom)
{

	GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();	
	
	Real dphidx = (phiOld(i+1,j,k)-phiOld(i-1,j,k))/(2*dx[0]);
	Real dphidy = (phiOld(i,j+1,k)-phiOld(i,j-1,k))/(2*dx[1]);
	Real modphisq = dphidx*dphidx + dphidy*dphidy;
	
	if(modphisq>1e-15){
		ac_val(i,j,k) = (1-3*dab) + 4*dab*(dphidx*dphidx*dphidx*dphidx+dphidy*dphidy*dphidy*dphidy)/(modphisq*modphisq);
	
	}
	
	else{
	
		ac_val(i,j,k) = 1.0;
		
	}
}

*/
///////////////////////////////////////////////////////////////////////////////////////////////////////
AMREX_GPU_DEVICE AMREX_FORCE_INLINE

void compute_ani(int i,int j,int k,amrex::Array4<amrex::Real const> const& phiOld, amrex::Array4<amrex::Real const> const& ac_val, amrex::Array4<amrex::Real> const& term1_val, amrex::Real gamma, amrex::Real dab, amrex::Geometry geom){
	
	GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

	
	Real dphidx_iph = (phiOld(i+1,j,k)-phiOld(i,j,k))/(dx[0]);
	Real dphidx_imh = (phiOld(i,j,k)-phiOld(i-1,j,k))/(dx[0]);
	Real dphidx_jph = (phiOld(i+1,j+1,k)-phiOld(i-1,j+1,k)+phiOld(i+1,j,k)-phiOld(i-1,j,k))/(4*dx[0]);
	Real dphidx_jmh = (phiOld(i+1,j,k)-phiOld(i-1,j,k)+phiOld(i+1,j-1,k)-phiOld(i-1,j-1,k))/(4*dx[0]);
	
	
	Real dphidy_jph = (phiOld(i,j+1,k)-phiOld(i,j,k))/(dx[1]);
	Real dphidy_jmh = (phiOld(i,j,k)-phiOld(i,j-1,k))/(dx[1]);
	Real dphidy_iph = (phiOld(i+1,j+1,k)-phiOld(i+1,j-1,k)+phiOld(i,j+1,k)-phiOld(i,j-1,k))/(4*dx[1]);
	Real dphidy_imh = (phiOld(i,j+1,k)-phiOld(i,j-1,k)+phiOld(i-1,j+1,k)-phiOld(i-1,j-1,k))/(4*dx[1]);
	

	Real modphisq_iph = dphidx_iph*dphidx_iph+dphidy_iph*dphidy_iph;
	Real modphisq_imh = dphidx_imh*dphidx_imh+dphidy_imh*dphidy_imh;
	Real modphisq_jph = dphidx_jph*dphidx_jph+dphidy_jph*dphidy_jph;
	Real modphisq_jmh = dphidx_jmh*dphidx_jmh+dphidy_jmh*dphidy_jmh;
	
	Real ac_iph, ac_jph, ac_imh, ac_jmh, acdashx_iph, acdashy_jph, acdashx_imh, acdashy_jmh;	
	
	if(modphisq_iph>1e-15){
	
	ac_iph = (1-3*dab) + 4*dab*(dphidx_iph*dphidx_iph*dphidx_iph*dphidx_iph+dphidy_iph*dphidy_iph*dphidy_iph*dphidy_iph)/(modphisq_iph*modphisq_iph);
	
	}
	
	else {
		ac_iph=1.0;
	}
	
	if(modphisq_jph>1e-15){
	
	ac_jph = (1-3*dab) + 4*dab*(dphidx_jph*dphidx_jph*dphidx_jph*dphidx_jph+dphidy_jph*dphidy_jph*dphidy_jph*dphidy_jph)/(modphisq_jph*modphisq_jph);
	
	}
	
	else {
		ac_jph=1.0;
	}
	
	if(modphisq_imh>1e-15){
	
	ac_imh = (1-3*dab) + 4*dab*(dphidx_imh*dphidx_imh*dphidx_imh*dphidx_imh+dphidy_imh*dphidy_imh*dphidy_imh*dphidy_imh)/(modphisq_imh*modphisq_imh);
	
	}
	
	else {
		ac_imh = 1.0;
	}
	
	if(modphisq_jmh>1e-15){
	
	ac_jmh = (1-3*dab) + 4*dab*(dphidx_jmh*dphidx_jmh*dphidx_jmh*dphidx_jmh+dphidy_jmh*dphidy_jmh*dphidy_jmh*dphidy_jmh)/(modphisq_jmh*modphisq_jmh);
	
	}
	
	else{
		ac_jmh = 1.0;
	}	
	
	//Real lap_phi = (phiOld(i+1,j,k)+phiOld(i,j+1,k)+phiOld(i,j-1,k)+phiOld(i-1,j,k)-phiOld(i,j,k))/(dx[0]*dx[0]);
	
	Real ani_term1 = 2.0*gamma*((ac_iph*ac_iph*dphidx_iph - ac_imh*ac_imh*dphidx_imh)/dx[0] + (ac_jph*ac_jph*dphidy_jph - ac_jmh*ac_jmh*dphidy_jmh)/dx[1]);

	if(modphisq_iph>1e-15){
	
	acdashx_iph = 16*dab*(dphidx_iph*dphidx_iph*dphidx_iph/(modphisq_iph)-dphidx_iph*(dphidx_iph*dphidx_iph*dphidx_iph*dphidx_iph+dphidy_iph*dphidy_iph*dphidy_iph*dphidy_iph)/(modphisq_iph*modphisq_iph));
	
	}
	
	else{
		acdashx_iph = 0.0;
	}
	
	if(modphisq_imh>1e-15){
	
	acdashx_imh = 16*dab*(dphidx_imh*dphidx_imh*dphidx_imh/(modphisq_imh)-dphidx_imh*(dphidx_imh*dphidx_imh*dphidx_imh*dphidx_imh+dphidy_imh*dphidy_imh*dphidy_imh*dphidy_imh)/(modphisq_imh*modphisq_imh));
	
	}
	
	else{
		acdashx_imh = 0.0;
	}
	
	if(modphisq_jph>1e-15){
	
	acdashy_jph = 16*dab*(dphidy_jph*dphidy_jph*dphidy_jph/(modphisq_jph)-dphidy_jph*(dphidx_jph*dphidx_jph*dphidx_jph*dphidx_jph+dphidy_jph*dphidy_jph*dphidy_jph*dphidy_jph)/(modphisq_jph*modphisq_jph));
	
	}
	
	else{
		acdashy_jph = 0.0;
	}
	
	if(modphisq_jmh>1e-15){
	
	acdashy_jmh = 16*dab*(dphidy_jmh*dphidy_jmh*dphidy_jmh/(modphisq_jmh)-dphidy_jmh*(dphidx_jmh*dphidx_jmh*dphidx_jmh*dphidx_jmh+dphidy_jmh*dphidy_jmh*dphidy_jmh*dphidy_jmh)/(modphisq_jmh*modphisq_jmh));
	
	}
	
	else{
		acdashy_jmh = 0.0;
	}
	

	
	Real ani_term2 = 2*gamma*((ac_iph*acdashx_iph-ac_imh*acdashx_imh)/(dx[0])+(ac_jph*acdashy_jph-ac_jmh*acdashy_jmh)/(dx[1]));
	
	term1_val(i,j,k) = ani_term1 + ani_term2;


}
///////////////////////////////////////////////////////////////////////////////////////////////////////

AMREX_GPU_DEVICE AMREX_FORCE_INLINE

void update_phi(int i,int j,int k, amrex::Array4<amrex::Real> const& phiNew, amrex::Array4<amrex::Real> const& phiOld, amrex::Array4<amrex::Real const> const& fin_term1, amrex::Array4<amrex::Real const> const& fin_term2, amrex::Array4<amrex::Real const> const& fin_term3, amrex::Real tau, amrex::Real dt, amrex::Real eps)
{	

	phiNew(i,j,k) = phiOld(i,j,k) + (dt/(tau*eps))*(eps*fin_term1(i,j,k)-fin_term2(i,j,k)/eps - fin_term3(i,j,k)/1.0e-5);
	
}
	
///////////////////////////////////////////////////////////////////////////////////////////////////////
// AMREX_GPU_DEVICE AMREX_FORCE_INLINE

// void update_temp(int i,int j,int k,amrex::Array4<amrex::Real const> const& phiOld,amrex::Array4<amrex::Real const> const& muOld,amrex::Array4<amrex::Real> const& muNew, amrex::Array4<amrex::Real const> const& phiNew, amrex::Real dt, amrex::Geometry const& geom)
// {
// 	GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

	
	
// }


//Function derivative
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void derivative_x(int i, int j, int k,
		amrex::Array4<amrex::Real const> const& phiOld,
		amrex::Array4<amrex::Real      > const& deriv,
		Geometry const& geom)
{
		GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();
		//Two sided difference
        deriv(i,j,k,iph) = (phiOld(i+1,j,k,0)-phiOld(i,j,k,0))/(dx[0]);
		deriv(i,j,k,imh) = (phiOld(i,j,k,0)-phiOld(i-1,j,k,0))/(dx[0]);
		deriv(i,j,k,jph) = 0.5*((phiOld(i+1,j+1,k,0)-phiOld(i-1,j+1,k,0))/(2.0*dx[0])+(phiOld(i+1,j,k,0)-phiOld(i-1,j,k,0))/(2.0*dx[0]));	//dPhi/dy - stored as component 1
		deriv(i,j,k,jmh) = 0.5*((phiOld(i+1,j,k,0)-phiOld(i-1,j,k,0))/(2.0*dx[0])+(phiOld(i+1,j-1,k,0)-phiOld(i-1,j-1,k,0))/(2.0*dx[0]));	//dPhi/dy - stored as component 1
		#if (AMREX_SPACEDIM > 2)
		deriv(i,j,k,kph) = 0.5*((phiOld(i+1,j,k+1,0)-phiOld(i-1,j,k+1,0))/(2.0*dx[0])+(phiOld(i+1,j,k,0)-phiOld(i-1,j,k,0))/(2.0*dx[0]));
		deriv(i,j,k,kmh) = 0.5*((phiOld(i+1,j,k,0)-phiOld(i-1,j,k,0))/(2.0*dx[0])+(phiOld(i+1,j,k-1,0)-phiOld(i-1,j,k-1,0))/(2.0*dx[0]));
		#endif

	//	Print(3)<<"("<<i<<","<<j<<","<<k<<") ";
	//	Print(3)<<deriv(i,j,k,iph)<<"\n";
		// AllPrint()<<deriv(i,j,k,imh)<<"\n";
		//Print(3)<<phiOld(i,j,k)<<"\n";
}

void derivative_y(int i, int j, int k,
		amrex::Array4<amrex::Real const> const& phiOld,
		amrex::Array4<amrex::Real      > const& deriv,
		Geometry const& geom)
{
		GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();
		//Two sided difference
        deriv(i,j,k,jph) = (phiOld(i,j+1,k,0)-phiOld(i,j,k,0))/(dx[1]);
		deriv(i,j,k,jmh) = (phiOld(i,j,k,0)-phiOld(i,j-1,k,0))/(dx[1]);
		deriv(i,j,k,iph) = 0.5*((phiOld(i+1,j+1,k,0)-phiOld(i+1,j-1,k,0))/(2.0*dx[1])+(phiOld(i,j+1,k,0)-phiOld(i,j-1,k,0))/(2.0*dx[1]));	//dPhi/dy - stored as component 1
		deriv(i,j,k,imh) = 0.5*((phiOld(i,j+1,k,0)-phiOld(i,j-1,k,0))/(2.0*dx[1])+(phiOld(i-1,j+1,k,0)-phiOld(i-1,j-1,k,0))/(2.0*dx[1]));	//dPhi/dy - stored as component 1
		#if (AMREX_SPACEDIM > 2)
		deriv(i,j,k,kph) = 0.5*((phiOld(i,j+1,k+1,0)-phiOld(i,j-1,k+1,0))/(2.0*dx[1])+(phiOld(i,j+1,k,0)-phiOld(i,j-1,k,0))/(2.0*dx[1]));
		deriv(i,j,k,kmh) = 0.5*((phiOld(i,j+1,k,0)-phiOld(i,j-1,k,0))/(2.0*dx[1])+(phiOld(i,j+1,k-1,0)-phiOld(i,j-1,k-1,0))/(2.0*dx[1]));
		#endif
}

#if (AMREX_SPACEDIM > 2)
void derivative_z(int i, int j, int k,
		amrex::Array4<amrex::Real const> const& phiOld,
		amrex::Array4<amrex::Real      > const& deriv,
		Geometry const& geom)
{
		GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();
		//Two sided difference
		deriv(i,j,k,jph) = 0.5*((phiOld(i,j+1,k+1,0)-phiOld(i,j+1,k-1,0))/(2.0*dx[2])+(phiOld(i,j,k+1,0)-phiOld(i,j,k-1,0))/(2.0*dx[2]));
		deriv(i,j,k,jmh) = 0.5*((phiOld(i,j,k+1,0)-phiOld(i,j,k-1,0))/(2.0*dx[2])+(phiOld(i,j-1,k+1,0)-phiOld(i,j-1,k-1,0))/(2.0*dx[2]));
		deriv(i,j,k,iph) = 0.5*((phiOld(i+1,j,k+1,0)-phiOld(i+1,j,k-1,0))/(2.0*dx[2])+(phiOld(i,j,k+1,0)-phiOld(i,j,k-1,0))/(2.0*dx[2]));	//dPhi/dy - stored as component 1
		deriv(i,j,k,imh) = 0.5*((phiOld(i,j,k+1,0)-phiOld(i,j,k-1,0))/(2.0*dx[2])+(phiOld(i-1,j,k+1,0)-phiOld(i-1,j,k-1,0))/(2.0*dx[2]));	//dPhi/dy - stored as component 1
		#if (AMREX_SPACEDIM > 2)
		deriv(i,j,k,kph) = (phiOld(i,j,k+1,0)-phiOld(i,j,k,0))/(dx[2]);
		deriv(i,j,k,kmh) = (phiOld(i,j,k,0)-phiOld(i,j,k-1,0))/(dx[2]);
		#endif
}
#endif



#endif	
	
	
	
