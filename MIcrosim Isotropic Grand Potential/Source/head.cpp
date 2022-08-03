#include "head.H"
#include "calc.H"

using namespace amrex;

void init_phi (MultiFab& phi_new, Real seed, Real ncell)
{
	for (MFIter mfi(phi_new); mfi.isValid(); ++mfi)
	{
		const Box& wbx = mfi.validbox();
		auto const& phiNew = phi_new.array(mfi);
		
		amrex::ParallelFor( wbx, [=] AMREX_GPU_DEVICE( int i, int j ,int k)
		{
			init_phi(i,j,k,phiNew,seed, ncell);
		});
	}
}	


void init_mu (MultiFab& mu_new, Real A_liq, Real ceq_liq)
{
	for (MFIter mfi(mu_new); mfi.isValid(); ++mfi)
	{
		const Box& wbx = mfi.validbox();
		auto const& muNew = mu_new.array(mfi);
		
		amrex::ParallelFor( wbx, [=] AMREX_GPU_DEVICE( int i, int j ,int k)
		{
			init_mu(i, j, k, muNew, A_liq, ceq_liq);
		});
	}
}	


			

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
		Geometry const& geom)
{
	//Fill the ghost cells
	phi_old.FillBoundary(geom.periodicity());		
	mu_old.FillBoundary(geom.periodicity());


	// for (MFIter mfi(phi_old); mfi.isValid(); ++mfi)
	// {
	// 	const Box& bbx = mfi.validbox();
	// 	auto const& phiOld = phi_old.const_array(mfi);
	// 	auto const& ac_val = ac.array(mfi);
		
	// 	amrex::ParallelFor( bbx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
	// 	{
	// 		compute_ac(i,j,k, phiOld, ac_val, dab, geom);
	// 	});
	// }
	
	

	//ac.FillBoundary(geom.periodicity());
	
	//Computing the anisotropy term(term1) in the phi evolution equation (Refer to calc.H for the formulation)

	for (MFIter mfi(phi_old); mfi.isValid(); ++mfi)
	{
		const Box& bbx = mfi.validbox();
		auto const& phiOld = phi_old.const_array(mfi);
		auto const& ac_val = ac.const_array(mfi);
		auto const& term1_val = term1.array(mfi);
		
		amrex::ParallelFor( bbx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
		{
		
		     compute_ani(i,j,k, phiOld, ac_val, term1_val, gamma, dab, geom);
		
		});
	}


	//Fill ghost cells for term1
	term1.FillBoundary(geom.periodicity());

	//Computing the Double well Potential(term2) in the phi evolution equation (Refer to head.cpp(below) for the formulation of double well potential calculation)
	computeterm2(term2, phi_old, gamma);

	//Fill ghost cells for term2
	term2.FillBoundary(geom.periodicity());

	//Computing the Psi equation(term3) in the phi evolution equation (Refer to head.cpp(below) for the formulation of psi calculation)
	computeterm3(mu_old, term3, phi_old, A, B, D);

	//Fill ghost cells for term3
	term3.FillBoundary(geom.periodicity());


	//Now we have all the terms for terms for phi evolution, we simply add them (Refer to calc.H for the formulation of update_phi function)
	for (MFIter mfi(phi_old); mfi.isValid(); ++mfi)
	{
		const Box& dbx = mfi.validbox();
		auto const& fin_term1 = term1.const_array(mfi);
		auto const& fin_term2 = term2.const_array(mfi);
		auto const& fin_term3 = term3.const_array(mfi);
		auto const& phiNew = phi_new.array(mfi);
		auto const& phiOld = phi_old.array(mfi);
		
		amrex::ParallelFor( dbx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
		{
		
			update_phi(i,j,k,phiNew, phiOld, fin_term1, fin_term2, fin_term3, tau, dt, eps);
		 	
		});
	}

	//Fill ghost cells of phi_new
	phi_new.FillBoundary(geom.periodicity());


	//Phi is already updated now here we update mu (Refer to dmudt fucntion in head.cpp(below) for formulation)
	dmudt(mu_new, mu_old, mu_der, phi_new, phi_old, diff, A, B, dt, geom);
	
}
	

void computeterm2(MultiFab& term2, MultiFab& phi_old, Real gamma)
{
	
	for ( MFIter mfi(phi_old); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
		auto const& phiold = phi_old.array(mfi);
		auto const& term = term2.array(mfi);
	
		amrex::ParallelFor(vbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
			term(i,j,k) = 9.0*gamma*2.0*phiold(i,j,k)*(1.0-phiold(i,j,k))*(1.0 - 2.0*phiold(i,j,k));
		});
	
	}
}


void computeterm3(MultiFab& mu_old, MultiFab& term3, MultiFab& phi_old, Vector<Real> A, Real B, Real D)
{
	

	for ( MFIter mfi(phi_old); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
		auto const& phiold = phi_old.array(mfi);
		auto const& term = term3.array(mfi);
		auto const& mu = mu_old.array(mfi);
		
	
		amrex::ParallelFor(vbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
			Real psia = -pow(((mu(i,j,k) - B)/(2.0*A[0])),2)*A[0] + D;
			Real psil = -pow(mu(i,j,k)/(2.0*A[1]),2)*A[1];

			term(i,j,k) = (6.0*phiold(i,j,k)*(1.0-phiold(i,j,k)))*(psia - psil);

		});
	
	}
}



void dmudt(MultiFab& mu_new, MultiFab& mu_old, Array<amrex::MultiFab, AMREX_SPACEDIM>& mu_der, MultiFab& phi_new, MultiFab& phi_old, Vector<Real> diffusivity, Vector<Real> A, Real B, Real dt, Geometry const& geom)
{
	GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

	Real dcadmu = 1.0/(2.0*A[0]);
	Real dcbdmu = 1.0/(2.0*A[1]);

	for ( MFIter mfi(phi_old); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
		auto const& phiold = phi_old.array(mfi);
		auto const& phinew = phi_new.array(mfi);
		auto const& mun = mu_new.array(mfi);
		auto const& muo = mu_old.array(mfi);

	
		amrex::ParallelFor(vbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
			Real calpha = (muo(i,j,k) - B)/(2.0*A[0]);
			Real cbeta = muo(i,j,k)/(2.0*A[1]);

			Real dmudx_iph = (muo(i+1,j,k)-muo(i,j,k))/(dx[0]);
			Real dmudx_imh = (muo(i,j,k)-muo(i-1,j,k))/(dx[0]);
			Real dmudy_jph = (muo(i,j+1,k)-muo(i,j,k))/(dx[1]);
			Real dmudy_jmh = (muo(i,j,k)-muo(i,j-1,k))/(dx[1]);

			Real dbdx = ((diffusivity[0]*0.5*(phiold(i,j,k)+phiold(i+1,j,k))*dcadmu + diffusivity[1]*(1.0 - 0.5*(phiold(i,j,k)+phiold(i+1,j,k)))*dcbdmu)*dmudx_iph
					-	 (diffusivity[0]*0.5*(phiold(i,j,k)+phiold(i-1,j,k))*dcadmu + diffusivity[1]*(1.0 - 0.5*(phiold(i,j,k)+phiold(i-1,j,k)))*dcbdmu)*dmudx_imh)/dx[0];

			Real dbdy = ((diffusivity[0]*0.5*(phiold(i,j,k)+phiold(i,j+1,k))*dcadmu + diffusivity[1]*(1.0 - 0.5*(phiold(i,j,k)+phiold(i,j+1,k)))*dcbdmu)*dmudy_jph
					-	 (diffusivity[0]*0.5*(phiold(i,j,k)+phiold(i,j-1,k))*dcadmu + diffusivity[1]*(1.0 - 0.5*(phiold(i,j,k)+phiold(i,j-1,k)))*dcbdmu)*dmudy_jmh)/dx[1];


			Real cdhdt = (calpha - cbeta)*(6.0*phiold(i,j,k)*(1.0-phiold(i,j,k)))*(phinew(i,j,k) - phiold(i,j,k))/dt;

			Real coeffdmudt = pow(phiold(i,j,k),2)*(3.0 - 2.0*phiold(i,j,k))*dcadmu + (1.0 - pow(phiold(i,j,k),2)*(3.0 - 2.0*phiold(i,j,k)))*dcbdmu;

			Real dmudt = (dbdx + dbdy - cdhdt)/coeffdmudt;

			mun(i,j,k) = muo(i,j,k) + dmudt*dt;

		});
	
	}
}








// // void dmudt(MultiFab& mu_new, MultiFab& mu_old, Array<amrex::MultiFab, AMREX_SPACEDIM>& mu_der, MultiFab& phi_new, MultiFab& phi_old, Vector<Real> diffusivity, Vector<Real> A, Real B, Real dt, Geometry const& geom)
// // {
// // 	GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

// // 	der(mu_old, mu_der, geom);
// // 	//Real V = 1.0e-5;
// // 	Real dcadmu = 1.0/(2.0*A[0]);
// // 	Real dcbdmu = 1.0/(2.0*A[1]);

// // 	for ( MFIter mfi(phi_old); mfi.isValid(); ++mfi )
// //     {
// //         const Box& vbx = mfi.validbox();
// //         auto const& der_x = mu_der[0].array(mfi);
// // 		auto const& der_y = mu_der[1].array(mfi);
// // 		#if (AMREX_SPACEDIM > 2)
// // 		auto const& der_z = mu_der[2].array(mfi);
// // 		#endif
// // 		auto const& phiold = phi_old.array(mfi);
// // 		auto const& phinew = phi_new.array(mfi);
// // 		auto const& mun = mu_new.array(mfi);
// // 		auto const& muo = mu_old.array(mfi);

	
// // 		amrex::ParallelFor(vbx,
// //         [=] AMREX_GPU_DEVICE (int i, int j, int k)
// //         {
// // 			Real calpha = (muo(i,j,k) - B)/(2.0*A[0]);
// // 			Real cbeta = muo(i,j,k)/(2.0*A[1]);


// // 			/*
// // 			Real phi_iph = (phiold(i+1,j,k)+phiold(i,j,k))/(2.0);
// // 			Real phi_imh = (phiold(i,j,k)+phiold(i-1,j,k))/(2.0);
	
	
// // 			Real phi_jph = (phiold(i,j+1,k)+phiold(i,j,k))/(2.0);
// // 			Real phi_jmh = (phiold(i,j,k)+phiold(i,j-1,k))/(2.0);


// // 			Real h_alpha_iph = 3.0*phi_iph*phi_iph-2.0*phi_iph*phi_iph*phi_iph;
// // 			Real h_alpha_imh = 3.0*phi_imh*phi_imh-2.0*phi_imh*phi_imh*phi_imh;
// // 			Real h_alpha_jph = 3.0*phi_jph*phi_jph-2.0*phi_jph*phi_jph*phi_jph;
// // 			Real h_alpha_jmh = 3.0*phi_jmh*phi_jmh-2.0*phi_jmh*phi_jmh*phi_jmh;


// // 			Real dbdx = ((diffusivity[0]*h_alpha_iph*dcadmu + diffusivity[1]*(1.0 - h_alpha_iph)*dcbdmu)*der_x(i,j,k,iph)
// // 					-	 (diffusivity[0]*h_alpha_imh*dcadmu + diffusivity[1]*(1.0 - h_alpha_imh)*dcbdmu)*der_x(i,j,k,imh))/dx[0];

// // 			Real dbdy = ((diffusivity[0]*h_alpha_jph*dcadmu + diffusivity[1]*(1.0 - h_alpha_jph)*dcbdmu)*der_y(i,j,k,jph)
// // 					-	 (diffusivity[0]*h_alpha_jmh*dcadmu + diffusivity[1]*(1.0 - h_alpha_jmh)*dcbdmu)*der_y(i,j,k,jmh))/dx[1];

// // 			*/


// // 			Real dbdx = ((diffusivity[0]*0.5*(phiold(i,j,k)+phiold(i+1,j,k))*dcadmu + diffusivity[1]*(1.0 - 0.5*(phiold(i,j,k)+phiold(i+1,j,k)))*dcbdmu)*der_x(i,j,k,iph)
// // 					-	 (diffusivity[0]*0.5*(phiold(i,j,k)+phiold(i-1,j,k))*dcadmu + diffusivity[1]*(1.0 - 0.5*(phiold(i,j,k)+phiold(i-1,j,k)))*dcbdmu)*der_x(i,j,k,imh))/dx[0];

// // 			Real dbdy = ((diffusivity[0]*0.5*(phiold(i,j,k)+phiold(i,j+1,k))*dcadmu + diffusivity[1]*(1.0 - 0.5*(phiold(i,j,k)+phiold(i,j+1,k)))*dcbdmu)*der_y(i,j,k,jph)
// // 					-	 (diffusivity[0]*0.5*(phiold(i,j,k)+phiold(i,j-1,k))*dcadmu + diffusivity[1]*(1.0 - 0.5*(phiold(i,j,k)+phiold(i,j-1,k)))*dcbdmu)*der_y(i,j,k,jmh))/dx[1];


// // 			Real cdhdt = (calpha - cbeta)*(6.0*phiold(i,j,k)*(1.0-phiold(i,j,k)))*(phinew(i,j,k) - phiold(i,j,k))/dt;

// // 			Real coeffdmudt = pow(phiold(i,j,k),2)*(3.0 - 2.0*phiold(i,j,k))*dcadmu + (1.0 - pow(phiold(i,j,k),2)*(3.0 - 2.0*phiold(i,j,k)))*dcbdmu;

// // 			Real dmudt = (dbdx + dbdy - cdhdt)/coeffdmudt;

// // 			mun(i,j,k) = muo(i,j,k) + dmudt*dt;

// // 		});
	
// // 	}
// // }


// void der(amrex::MultiFab& phiold, Array<amrex::MultiFab, AMREX_SPACEDIM>& derivative, amrex::Geometry const& geom)
// {
// 	for ( MFIter mfi(phiold); mfi.isValid(); ++mfi )
// 	{
// 		const Box& vbx = mfi.validbox();
//         auto const& phiOld = phiold.array(mfi);
// 		auto const& der_x = derivative[0].array(mfi);
// 		auto const& der_y = derivative[1].array(mfi);
// 		#if(AMREX_SPACEDIM > 2)
// 		auto const& der_z = derivative[2].array(mfi);
// 		#endif
        
// 		amrex::ParallelFor(vbx,
//         [=] AMREX_GPU_DEVICE (int i, int j, int k)
// 		{
// 			derivative_x(i,j,k, phiOld, der_x, geom);
// 			derivative_y(i,j,k, phiOld, der_y, geom);
// 			#if (AMREX_SPACEDIM > 2)
// 			derivative_z(i,j,k,phiOld,der_z,geom);
// 			#endif

// 		});
// 	}
// }

// void calc_eeta(Array<MultiFab, AMREX_SPACEDIM>& derivative, MultiFab& eeta, Real delta)
// {
	
// 	for ( MFIter mfi(eeta); mfi.isValid(); ++mfi )
//     {
//         const Box& vbx = mfi.validbox();
//         auto const& der_x = derivative[0].array(mfi);
// 		auto const& der_y = derivative[1].array(mfi);
// 		#if (AMREX_SPACEDIM > 2)
// 		auto const& der_z = derivative[2].array(mfi);
// 		#endif
// 		auto const& eta = eeta.array(mfi);
	
// 		amrex::ParallelFor(vbx,
//         [=] AMREX_GPU_DEVICE (int i, int j, int k)
//         {
// 				eta(i,j,k, iph) = (pow((pow(der_x(i,j,k,iph),2)+pow(der_y(i,j,k,iph),2)),2) > 1.0e-15) ? (1.0 - 3.0*delta + 4.0*delta*(pow(der_x(i,j,k,iph),4)+pow(der_y(i,j,k,iph),4))/pow((pow(der_x(i,j,k,iph),2)+pow(der_y(i,j,k,iph),2)),2)) : 1.0;/*(1.0 - 3.0*delta);*/
// 				eta(i,j,k, imh) = (pow((pow(der_x(i,j,k,imh),2)+pow(der_y(i,j,k,imh),2)),2) > 1.0e-15) ? (1.0 - 3.0*delta + 4.0*delta*(pow(der_x(i,j,k,imh),4)+pow(der_y(i,j,k,imh),4))/pow((pow(der_x(i,j,k,imh),2)+pow(der_y(i,j,k,imh),2)),2)) : 1.0;//(1.0 - 3.0*delta);
// 				eta(i,j,k, jph) = (pow((pow(der_x(i,j,k,jph),2)+pow(der_y(i,j,k,jph),2)),2) > 1.0e-15) ? (1.0 - 3.0*delta + 4.0*delta*(pow(der_x(i,j,k,jph),4)+pow(der_y(i,j,k,jph),4))/pow((pow(der_x(i,j,k,jph),2)+pow(der_y(i,j,k,jph),2)),2)) : 1.0;//(1.0 - 3.0*delta);
// 				eta(i,j,k, jmh) = (pow((pow(der_x(i,j,k,jmh),2)+pow(der_y(i,j,k,jmh),2)),2) > 1.0e-15) ? (1.0 - 3.0*delta + 4.0*delta*(pow(der_x(i,j,k,jmh),4)+pow(der_y(i,j,k,jmh),4))/pow((pow(der_x(i,j,k,jmh),2)+pow(der_y(i,j,k,jmh),2)),2)) : 1.0;//(1.0 - 3.0*delta);
			

// 		/*	if (pow((pow(der_x(i,j,k,iph),2)+pow(der_y(i,j,k,iph),2)),2) > 1.0e-5)
// 			{
// 				eta(i,j,k, iph) = 1 - 3*delta + 4*delta*(pow(der_x(i,j,k,iph),4)+pow(der_y(i,j,k,iph),4))/pow((pow(der_x(i,j,k,iph),2)+pow(der_y(i,j,k,iph),2)),2);
// 				eta(i,j,k, imh) = 1 - 3*delta + 4*delta*(pow(der_x(i,j,k,imh),4)+pow(der_y(i,j,k,imh),4))/pow((pow(der_x(i,j,k,imh),2)+pow(der_y(i,j,k,imh),2)),2);
// 				eta(i,j,k, jph) = 1 - 3*delta + 4*delta*(pow(der_x(i,j,k,jph),4)+pow(der_y(i,j,k,jph),4))/pow((pow(der_x(i,j,k,jph),2)+pow(der_y(i,j,k,jph),2)),2);
// 				eta(i,j,k, jmh) = 1 - 3*delta + 4*delta*(pow(der_x(i,j,k,jmh),4)+pow(der_y(i,j,k,jmh),4))/pow((pow(der_x(i,j,k,jmh),2)+pow(der_y(i,j,k,jmh),2)),2);
// 			}
// 			else
// 			{
// 				eta(i,j,k, iph) = 1 - 3*delta;
// 				eta(i,j,k, imh) = 1 - 3*delta;
// 				eta(i,j,k, jph) = 1 - 3*delta;
// 				eta(i,j,k, jmh) = 1 - 3*delta;
// 			}
// 		*/


// 	//		Print(3)<<"("<<i<<", "<<j<<", "<<k<<") = ";
// 	//		Print(3)<<"Eta : "<<eta(i,j,k,iph)<<"\n"; 

// 		});
       
//     }
// }


// void anisotropy (MultiFab& eeta,
//                 MultiFab& term1,
// 				Array<amrex::MultiFab,AMREX_SPACEDIM>& derivative,
// 				Real delta,
// 				Real gamma,
// 				Geometry const& geom)
// {
// 	//dx array stores the cell size in each dimension
// 	GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

//     //Iterate over the entire fabs to update phi and temperature     
//     for ( MFIter mfi(eeta); mfi.isValid(); ++mfi )
//     {
//         const Box& vbx = mfi.validbox();
// 		auto const& eta = eeta.array(mfi);
//         auto const& ani = term1.array(mfi);
// 		//	auto const& eps = epsilon.array(mfi);
// 		auto const& der_x = derivative[0].array(mfi);
// 		auto const& der_y = derivative[1].array(mfi);
// 		#if(AMREX_SPACEDIM > 2)
// 		auto const& der_z = derivative[2].array(mfi);
// 		#endif
	
// 		//Print(2)<<"0\n";
// 		amrex::ParallelFor(vbx,
//         [=] AMREX_GPU_DEVICE (int i, int j, int k)
//         {		
		    
// 		    Real xterm, yterm, div_term;

	
// 			xterm = (((pow((pow(der_x(i,j,k,iph),2)+pow(der_y(i,j,k,iph),2)),2)) > 1.0e-15) ? (eta(i,j,k,iph)*16*delta/(pow((pow(der_x(i,j,k,iph),2)+pow(der_y(i,j,k,iph),2)),2))*  der_x(i,j,k,iph) * ( pow(der_x(i,j,k,iph),2) * (pow(der_x(i,j,k,iph),2)+pow(der_y(i,j,k,iph),2))-(pow(der_x(i,j,k,iph),4)-pow(der_y(i,j,k,iph),4)))):0.0
// 				   - ((pow((pow(der_x(i,j,k,imh),2)+pow(der_y(i,j,k,imh),2)),2)) > 1.0e-15) ? (eta(i,j,k,imh)*16*delta/(pow((pow(der_x(i,j,k,imh),2)+pow(der_y(i,j,k,imh),2)),2))*  der_x(i,j,k,imh) * ( pow(der_x(i,j,k,imh),2) * (pow(der_x(i,j,k,imh),2)+pow(der_y(i,j,k,imh),2))-(pow(der_x(i,j,k,imh),4)-pow(der_y(i,j,k,imh),4)))):0.0)/dx[0];
			
// 			yterm = (((pow((pow(der_x(i,j,k,jph),2)+pow(der_y(i,j,k,jph),2)),2)) > 1.0e-15) ? (eta(i,j,k,jph)*16*delta/(pow((pow(der_x(i,j,k,jph),2)+pow(der_y(i,j,k,jph),2)),2))*  der_y(i,j,k,jph) * ( pow(der_y(i,j,k,jph),2) * (pow(der_x(i,j,k,jph),2)+pow(der_y(i,j,k,jph),2))-(pow(der_x(i,j,k,jph),4)-pow(der_y(i,j,k,jph),4)))):0.0
// 				   - ((pow((pow(der_x(i,j,k,jmh),2)+pow(der_y(i,j,k,jmh),2)),2)) > 1.0e-15) ? (eta(i,j,k,jmh)*16*delta/(pow((pow(der_x(i,j,k,jmh),2)+pow(der_y(i,j,k,jmh),2)),2))*  der_y(i,j,k,jmh) * ( pow(der_y(i,j,k,jmh),2) * (pow(der_x(i,j,k,jmh),2)+pow(der_y(i,j,k,jmh),2))-(pow(der_x(i,j,k,jmh),4)-pow(der_y(i,j,k,jmh),4)))):0.0)/dx[1];

// 			div_term = ((eta(i,j,k,iph)*eta(i,j,k,iph)*der_x(i,j,k,iph) - eta(i,j,k,imh)*eta(i,j,k,imh)*der_x(i,j,k,imh))/dx[0]  +  (eta(i,j,k,jph)*eta(i,j,k,jph)*der_y(i,j,k,jph) - eta(i,j,k,jmh)*eta(i,j,k,jmh)*der_y(i,j,k,jmh))/dx[1]);
		
	

// 		    Real laplacian_term = xterm + yterm + div_term;

//             ani(i,j,k) = 2.0 * gamma*laplacian_term;
        
// 		});
       
//     }

// 	/*	for ( MFIter mfi(phi_new); mfi.isValid(); ++mfi )               //DEBUG
// 		{
// 			const Box& vbx = mfi.validbox();	
// 			auto const& phiNew = phi_new.array(mfi);
			
		
// 			amrex::ParallelFor(vbx,
// 			[=] AMREX_GPU_DEVICE (int i, int j, int k)
// 			{
// 				Print(3)<<"("<<i<<","<<j<<","<<k<<") ";
			
// 				Print(3)<<"phi = "<<phiNew(i,j,k)<<"\n";
// 			});
// 		}
// 	*/

// }


void calc_h(MultiFab& phi_new, MultiFab& h)
{
	
	for ( MFIter mfi(phi_new); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
        auto const& hphi = h.array(mfi);
		auto const& phi = phi_new.array(mfi);
	
		amrex::ParallelFor(vbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
			hphi(i,j,k) = 3.0*phi(i,j,k)*phi(i,j,k) - 2.0*phi(i,j,k)*phi(i,j,k)*phi(i,j,k); 

		});
       
    }
}