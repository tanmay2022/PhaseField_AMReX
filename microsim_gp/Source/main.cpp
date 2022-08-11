#include <AMReX_Gpu.H>
#include <AMReX_Utility.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include "head.H"

using namespace amrex;

int main( int argc, char* argv[])
{
	amrex::Initialize(argc, argv);
	
	GPotential();
	
	amrex::Finalize();
	return 0;
	
}


void GPotential()
{
	auto strt_time = ParallelDescriptor::second();
	int ncell, maxgrid, nsteps, plotint;
	Real eps, dab, gamma, seed, dt, Afill, real_bound,Vm;
	Vector<amrex::Real> A(2,0);
	Vector<amrex::Real> Aeq(2,0);
	Vector<amrex::Real> ceq(2,0);
	Vector<amrex::Real> cuc(2,0);
	Vector<amrex::Real> diff(2,0);

	{
		ParmParse pp;
		
		pp.get("real_bound", real_bound);
		pp.get("ncell",ncell);
		pp.get("maxgrid",maxgrid);
		pp.get("nsteps",nsteps);
		pp.get("plotint",plotint);
		pp.get("epsilon",eps);
		pp.get("dab",dab);
		pp.get("gamma",gamma);
		pp.get("A_fill",Afill);
		pp.get("seed",seed);
		pp.get("dt",dt);
		pp.get("Vm",Vm);

		pp.queryarr("A", A);
		pp.queryarr("Aeq", Aeq);
		pp.queryarr("ceq", ceq);
		pp.queryarr("cuc", cuc);
		pp.queryarr("diffusivity", diff);
	}
	
	BoxArray ba;
	Geometry geom;
	{
		IntVect dom_lo(AMREX_D_DECL(0,0,0));
		IntVect dom_high(AMREX_D_DECL(ncell-1,ncell-1,ncell-1));
		Box domain(dom_lo,dom_high);
		ba.define(domain);
		ba.maxSize(maxgrid);
		
		RealBox real_box({AMREX_D_DECL(Real(0.0),Real(0.0),Real(0.0))},
				{AMREX_D_DECL(real_bound, real_bound, real_bound)});
	 	Array<int, AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(1,1,1)};
	 	geom.define(domain, real_box, CoordSys::cartesian, is_periodic);
	 	
	 }
	 

	 int ghost=1;
	 int comp=1;
	 
	 DistributionMapping dm(ba);
	 
	 MultiFab phi_old(ba, dm, comp, ghost);
	 MultiFab phi_new(ba, dm, comp, ghost);
	 MultiFab mu_old(ba, dm, comp, ghost);
	 MultiFab mu_new(ba, dm, comp, ghost);
	 Array<amrex::MultiFab, AMREX_SPACEDIM> deriv;
	 Array<amrex::MultiFab, AMREX_SPACEDIM> mu_der;
	 

	 MultiFab ac(ba, dm, comp, ghost);
	 MultiFab term1(ba, dm, comp, ghost);
	 MultiFab term2(ba, dm, comp, ghost);
	 MultiFab term3(ba, dm, comp, ghost);
	 MultiFab eeta(ba, dm, 4, 0);
	 MultiFab print(ba,dm,6,1);
	 
	 phi_old.setVal(0.0);
	 phi_new.setVal(0.0);
	 mu_old.setVal(0.0);
	 mu_new.setVal(0.0);
	// for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
    // {
    //     BoxArray boxarray = ba;
    //     #if (AMREX_SPACEDIM > 2)
    //     mu_der[dir].define(boxarray, dm, 6, 0);
    //     mu_der[dir].setVal(0.0);
	// 	deriv[dir].define(boxarray, dm, 6, 0);
    //     deriv[dir].setVal(0.0);
    //     #else
    //     mu_der[dir].define(boxarray, dm, 4, 0);
    //     mu_der[dir].setVal(0.0);
	// 	deriv[dir].define(boxarray, dm, 4, 0);
    //     deriv[dir].setVal(0.0);
    //     #endif
    // }
	 ac.setVal(0.0);
	 term1.setVal(0.0);
	 term2.setVal(0.0);
	 term3.setVal(0.0);

	Afill=Afill/2.0; 
	A[0]=A[0]/2.0;
	A[1]=A[1]/2.0; 
	Aeq[0]=Aeq[0]/2.0;
	Aeq[1]=Aeq[1]/2.0; 
	
	Real B = 2.0*A[1]*cuc[1] - 2.0*A[0]*cuc[0];
	Real D = A[0]*cuc[0]*cuc[0] - A[1]*cuc[1]*cuc[1];

	Real tau = 0.222*eps*(ceq[0]-ceq[1])*(ceq[0]-ceq[1])*2*Aeq[1]/(diff[1]*Vm);
	
	

	 GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
	 
	 init_phi(phi_new, seed, ncell);
	 init_mu(mu_new, Aeq[1], ceq[1]);
 

	 MultiFab h(ba, dm, 1, 0);
	 calc_h(phi_new, h);
	 
	 Real time = 0.0;
	 
	 MultiFab::Copy(print, phi_new, 0, 0, 1, 0);
	 MultiFab::Copy(print, term1, 0, 1, 1, 0);
	 MultiFab::Copy(print, term2, 0, 2, 1, 0);
	 MultiFab::Copy(print, term3, 0, 3, 1, 0);
	 MultiFab::Copy(print, mu_new, 0, 4, 1, 0);
	 MultiFab::Copy(print, h, 0, 5, 1, 0);

	 if(plotint>0)
	 {
	 	const std::string& pltfile  = amrex::Concatenate("plt",0,5);
	 	WriteSingleLevelPlotfile(pltfile, print, {"phi","term1","term2","term3","mu_new","h"},geom,time,0);
	 }
		Print()<<"B = "<<B<<"\n";
	 	Print()<<"D = "<<D<<"\n";

		Print()<<"dx = "<<dx[0]<<"\n";
		Print()<<"h_max = "<<h.max(0,0,0)<<"\n";
		Print()<<"h_min = "<<h.min(0,0,0)<<"\n";
		Print()<<"mu_max = "<<mu_new.max(0,0,0)<<"\n";
		Print()<<"mu_min = "<<mu_new.min(0,0,0)<<"\n";
	 	Print()<<"phi_max = "<<phi_new.max(0,0,0)<<"\n";
		Print()<<"phi_min = "<<phi_new.min(0,0,0)<<"\n";

	 
	 for(int n=1; n<=nsteps; ++n)
	 {
	 	MultiFab::Copy(phi_old, phi_new, 0,0,1,0);
	 	MultiFab::Copy(mu_old, mu_new, 0,0,1,0);
	 	
	 	advance(phi_old, phi_new, deriv, mu_old, mu_new, mu_der, term1, term2, term3, ac, eeta, gamma, dab, tau, dt, eps, A, B, D, diff, Vm, geom);
	 	
	 	time=time+dt;
	 	calc_h(phi_new, h);

		Print()<<"1_max = "<<term1.max(0,0,0)*eps<<"\n";
		Print()<<"1_min = "<<term1.min(0,0,0)*eps<<"\n";
		Print()<<"2_max = "<<term2.max(0,0,0)/eps<<"\n";
		Print()<<"2_min = "<<term2.min(0,0,0)/eps<<"\n";
		Print()<<"3_max = "<<term3.max(0,0,0)/1.0e-5<<"\n";
		Print()<<"3_min = "<<term3.min(0,0,0)/1.0e-5<<"\n";

		amrex::Print()<<"Advanced step"<<n<<"\n";

		Print()<<"h_max = "<<h.max(0,0,0)<<"\n";
		Print()<<"h_min = "<<h.min(0,0,0)<<"\n";
		Print()<<"mu_max = "<<mu_new.max(0,0,0)<<"\n";
		Print()<<"mu_min = "<<mu_new.min(0,0,0)<<"\n";
	 	Print()<<"phi_max = "<<phi_new.max(0,0,0)<<"\n";
		Print()<<"phi_min = "<<phi_new.min(0,0,0)<<"\n";


		MultiFab::Copy(print, phi_new, 0, 0, 1, 0);
		MultiFab::Copy(print, term1, 0, 1, 1, 0);
		MultiFab::Copy(print, term2, 0, 2, 1, 0);
		MultiFab::Copy(print, term3, 0, 3, 1, 0);
		MultiFab::Copy(print, mu_new, 0, 4, 1, 0);
		MultiFab::Copy(print, h, 0, 5, 1, 0);
	 	
	 	if(plotint>0 && n%plotint==0)
	 	{
	 		const std::string& pltfile = amrex::Concatenate("plt",n,5);
	 		WriteSingleLevelPlotfile( pltfile, print, {"phi","term1","term2","term3","mu_new","h"},geom,time,n);
	 	}
	 	
	 	
	 	// if(plotint>0 && n%plotint==0)
	 	// {
	 	// 	const std::string& pltfile = amrex::Concatenate("plt",n,5);
	 	// 	WriteSingleLevelPlotfile( pltfile, phi_new, {"phi"},geom,time,n);
	 	// }
	 	
	 	
	 	
	 }
	 
	 auto stop_time = ParallelDescriptor::second()-strt_time;
	 const int IOProc = ParallelDescriptor::IOProcessorNumber();
	 ParallelDescriptor::ReduceRealMax(stop_time, IOProc);
	 
	 amrex::Print()<<"Run time = "<<stop_time<<"\n";
	 
}
	 		
	 
