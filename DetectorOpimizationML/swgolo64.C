///////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                           //   
//                               S  W  G  O  L  O                                            //
//                                                                                           //   
//               Optimization of the footprint of SWGO detector array                        //
//               ----------------------------------------------------                        //
//                                                                                           //   
//  We use a quick and dirty parametrization of muon and electron+gamma fluxes as a          //
//  function of radius R for energetic air showers, and a simplified description             //
//  of detector units in terms of efficiency and acceptance, to estimate the                 //
//  uncertainty on primary gamma fluxes for different energies E, combining them in a        //
//  utility function as a function of detector positions on the ground.                      //
//                                                                                           //
//  This version of the code fits both the position of showers (x0,y0) and their             //
//  polar and azimuthal angles theta, phi and energy E through a likelihood maximization.    //
//  The fit is performed twice - once for the gamma and once for the proton hypothesis.      //
//  The two values of logL at maximum are used in the construction of a likelihood-ratio TS. // 
//  The distribution of this TS for the two hypotheses is the basis of the extraction        //
//  of the uncertainty on the signal fraction in the shower batches, with which a utility    //
//  value can be computed as f_s/sigma_f_s. Derivatives allow gradient descent to better     //
//  layouts of the array of detectors.                                                       //       
//                                                       T. Dorigo, 2022 - 2023              //
//                                                                                           //
//  Instructions to build and run project:                                                   //
//  --------------------------------------                                                   //
//  0. Create a directory for the project, e.g. /user/pippo/swgo/                            //
//  1. Copy this file to that directory                                                      //
//  2. Create subdirectories:                                                                //
//     - /user/pippo/swgo/Layouts     --> will contain plots of evolving config during run   //
//     - /user/pippo/swgo/Model       --> contains two txt files with model parameters       //
//     - /user/pippo/swgo/Dets        --> contains txt files with detector configurations    //
//     - /user/pippo/swgo/Outputs     --> contains dumps from runs of the program            //
//     - /user/pippo/swgo/Root        --> contains root files in output                      //
//  3. Copy model files into ./Model                                                         //
//     - Fit_Photon_10_pars.txt       --> parameters of photon model                         //
//     - Fit_Proton_2_pars.txt        --> parameters of proton model                         //
//  4. Copy detector layout files to ./Dets directory (optional)                             //
//  5. Modify this code to include the correct path to the main directory:                   //
//     Search and replace the string "/lustre/cmswork/dorigo/swgo/MT/"                       //
//                   with the string "/user/pippo/swgo/"                                     //
//  6. Compile with:                                                                         //
//     > g++ -g swgolo64.C `root-config --libs --cflags` -o swgolo64                         //
//     or if using derivgrind:                                                               //  
//     > g++ -g swgolo59.C -D WITH_DERIVGRIND `root-config --libs --cflags` -o swgolo59      //
//     In that case first also do                                                            //
//     > source Export_derivgrind.sh                                                         //
//     (sets up paths)                                                                       //
//     and run it with                                                                       //
//     > valgrind --tool=derivgrind ./swgolo59                                               //
//     (step 7)                                                                              //
//  7. Run it by specifying all parameters you wish to test. E.g.:                           //
//     > nohup ./swgolo59 -nev 2000 -nba 2000 -nde 100 -nep 500 -sha 3 -nth 20               //
//                                                                                           //
//     Note, the code is slow - the above parameters will produce one iteration              //
//     in several minutes on a single CPU. with -nth you can speed this up almost            //
//     by the number of CPUs your system can deploy.                                         //
//                                                                                           //                                                                                          //
//  Extensions to be worked at (as of August 14 2023):                                       //
//  --------------------------------------------------                                       //
//  - Implement choice of initial layouts as per SWGO baselines (one is in, 101)             //
//  - Treat separately electron/positron and gammas on the ground                            // 
//  - Introduce particle arrival time separately for e/gamma and mu                          //
//  - Create valid model of secondaries arrival time (for now using flat arrival profile)    //
//  - Implement efficiency for each particle type as integrated measure (indep of Epart)     // 
//  - Implement fake rate (spurious muons particularly can affect performance)               //
//  - Implement resolution function per tank in Npart detection                              //
//  - Implement distribution of particle energies and E dependent efficiency                 //
//  - Implement calculation of correct flux through physical tanks in flux calculations      //
//    (for now the flux is computed at center of the tank only)                              //
//  - Implement more realistic model of fluxes versus polar angle of primary                 //
//  - Perhaps voiding showers too close to a det during calc of dUdx may help convergence?   //
//                                                                                           //
//                                                                                           //
//  Pending fixes (as of August 14 2023):                                                    //
//  ---------------------------------                                                        //  
//  - Add contribution of dPActive/dx to U_IR component in calculation of dU/dx - done       //
//  - Improve likelihood fitting routine, particularly ADAM procedure. -> done?              //
//  - Optimize code for speed -> some done                                                   //
//  - Check all calculations. Tedious! -> in progress                                        //
//  - Fix derivatives wrt theta and phi due to new model (e.g. in dlogLRdR) -> done?         //
//  - Fix indfile issue when readgeom is true (adding epochs)                                //
//                                                                                           //
//  Updates:                                                                                 //
//  --------                                                                                 //
//  v56  version with preprocessing directive allowing switch between root and standalone    //
//  v56w fixed Poisson calculation in ProbTrigger routine                                    //
//  v57  changed U_IR and derivatives                                                        //
//  v58  various small fixes, file indexing                                                  //
//  v59  changed again U_IR to give less weight to very large fluke E measurements           //
//  v60  added missing dtheta/dr dependence in loglr_dr routine; fixed some bugs             //
//  v61  renewed calculation of dpg/dx, dpp/dx                                               //
//  v62  corrected a nasty bug dlrdx->dy, a minus sign in ProbTrigger, and added             //
//       calculation of measured gamma fraction (had been using Asimov Fg earlier)           //
//  v63  Now dlogLR_dr passes back a pair (derivatives wrt x,y) speeding up calcs.           //
//       Also, cosdir plots show acos of angle (want it flattish?)                           //
//  v64  Change dynamically the area where showers are generated                             //                                                                                         //
///////////////////////////////////////////////////////////////////////////////////////////////

// Please take care, no spaces before preprocessing directives, or they count as indentations/matching!
//#define PLOTS
#define FEWPLOTS
// use one of these alternatively, not both!
#define INROOT 
//#define STANDALONE
// ditto for the two above

#include "TH2.h"
#include "TH1.h"
#include "TF1.h"
#include "TProfile.h"
#include "TStyle.h"
#include "TCanvas.h" 
#include "TROOT.h"
#include "TFile.h"
#include "TMath.h"
//#include "Tmatrix.h"
#include <math.h>
#include "TRandom.h"
#include "TRandom3.h"
#include "Riostream.h"

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>

#ifdef STANDALONE
// For multithreading instructions
//#include <iostream>
#include <thread>
#include <mutex>

// Define a mutex for synchronizing access to shared data
// Note, this is not presently used as the threading functions do not
// access shared variables.
// ------------------------------------------------------------------
std::mutex dataMutex;
#endif

using namespace std;

// UNITS: 
// ------
// E in PeV
// length in meters
// time in nanoseconds
// angles in radians

// Constants and control settings
// ------------------------------ 
static const double largenumber    = 10000000000000.;
static const double epsilon        = 1./largenumber;
static const double epsilon2       = epsilon*epsilon;
static const double c0             = 0.29979;   // Speed of light in m/ns
static const double pi             = 3.1415926;
static const double twopi          = 2.*pi;
static const double halfpi         = 0.5*pi;
static const double sqrt2          = sqrt(2.);
static const double sqrt2pi        = sqrt(twopi);
static const double log_10         = log(10.);
static const double log_01         = log(0.1);
static const double logdif         = log_10-log_01;
static const double thetamax       = pi*65./180.;
static const double beta1          = 0.85;      // Parameter of ADAM optimizer. Default is 0.9
static const double beta2          = 0.95;      // Parameter of ADAM optimizer. Default is 0.999
static const bool   debug          = false;     // If on, the code generates lots of printouts; use with caution. 
static const bool   plotdistribs   = false;     // If on, plot densities per m^2 of shower profiles (the models from which Nmu, Ne are drawn)
static const bool   checkmodel     = false;     // If on, plot model distributions at the start
static const int    initBitmap     = 6;         // Binary map to initialize parameters of showers to their true values. 1=E, 2=P, 4=T, 8=Y0, 16=X0
static bool         usetrueXY      = false;     // If on, avoids fitting for x0,y0 of shower; NB don't have both usetrueXY and usetrueAngs on!
static const bool   SetTo00        = false;     // If on, all showers hit the center of the array
static bool         usetrueAngs    = false;     // If on, avoids fitting for theta, phi of shower
static bool         usetrueE       = false;     // If on, avoids fitting for E of shower
static       bool   fixShowerPos   = false;     // If on, showers are always generated at the same locations - reduces stochasticity 
static bool         OrthoShowers   = false;     // If on, showers come down orthogonally to the ground (theta=0)
static       bool   SlantedShowers = false;     // If on, showers come in at theta=pi/4
static const bool   hexaShowers    = true;      // Used if fixShowerPos is true, to set the geometry of showers x,y centers
static const bool   SameShowers    = false;     // If on, the same showers are generated at each epoch (for debugging only)
static const bool   addSysts       = false;     // If on, we mess up a bit the distributions to mimic an imperfect knowledge of the models // not developed yet
static const bool   checkUtility   = false;     // If on, the utility function is recomputed after a detector move, to check increase // under development
static       bool   scanU          = false;  // If on, U is scanned as a function of position of a detector unit
static const int    idstar         = 10;         // Id of det for which gradient is computed, see scanU
static bool         readGeom       = false;     // If on, reads detector positions from file. Fix also startEpoch if true.
static       int    startEpoch     = 0;         // Starting epoch, for cases when continuing a run with readGeom true
static const bool   writeGeom      = true;      // If on, writes final detector positions to file
static int          indfile;                    // This index contains the current id of the run for these parameters
static const int    SampleT        = false;     // Whether sigmaLRT is sampled to get its variance
static const int    Nrep           = 10;        // Number of repetitions of T evaluation for determination of sigmaLRT

// Max dimensioning constants
// --------------------------
#ifdef STANDALONE
static const int    maxUnits       = 8371;      // Max number of detectors that can be deployed 
#endif
#ifdef INROOT
static const int    maxUnits       = 1000;      // Max number of detectors that can be deployed 
#endif
static const int    minUnits       = 10;        // Lower limit on number of detectors
static const int    maxEvents      = 6000;      // Max events simulated for templates
static const int    maxEpochs      = 5000;      // Max number of epochs of utility maximization
static const int    maxRbins       = 100;       // Max number of bins in R where to average utility (if commonMode=1)
static const int    maxEinit       = 50;        // Max number of energy points for initial assay of shower likelihood
static const int    maxSteps       = 500;       // Max number of steps in likelihood maximization
static const int    maxNtrigger    = 60;        // Max number of required units seeing a signal for triggering
static int          maxNsteps      = 500.;      // Max steps in likelihood maximization. 300 works fine with adam optimizer (see below)

// Global parameters required for dimensioning arrays, and defaults
// ----------------------------------------------------------------
static int Nunits                  = 500;       // Number of detector units
static int Nbatch                  = 2000;      // Number of showers per batch
static int Nevents                 = 2000;      // Number of showers to determine shape of TS at each step
static int Nepochs                 = 100;       // Number of SGD cycles performed
static int NRbins                  = 100;       // Number of R bins in xy plane, to average derivatives of U (only relevant for commonMode=1)
static int Nthreads                = 25;        // Number of threads for cpu-intensive routines
static int N_predef[13]            = { 6589, 6631, 6823, 6625, 6541, 6637, 6571, 4849, 8371, 3805, 5461, 5455, 4681 };

// Other parameters defining run
// -----------------------------
static double spanR                = 0.;        // Span in R of detector array. This gets defined based on the initial layout of x[], y[]
static double Xoffset              = 0.;        // Used to study behaviour of maximization and "drift to interesting region"
static double Yoffset              = 0.;        // Same as above for y: showers are "off center" with respect to the region where detectors are
static double XDoffset             = 0.;        // With these we put detector units offset instead of showers
static double YDoffset             = 0.;        // Same as above for y
static double Rmin                 = 0.1;       // Minimum considered distance of shower center from detector (avoiding divergence of gradients) 
static int commonMode              = 0;         // Choice to vary all xy (0), R (1), or common center (2) of array during SGD
static double MaxUtility           = 100.;      // max value of plotted utility in U graph
static double TankArea             = 25.27*pi;  // 25.27pi for 7 1.9m radius "macro-tanks" aggregates in hexagonal pattern (works better than smaller units)
static double StartLR              = 1.0;       // Initial learning rate in SGD
static double eta_GF               = 0.1;       // Weight of gamma fraction in utility
static double eta_IR               = 10.;       // Weight of integrated resolution in utility
static double U_IR_Num;
static double U_IR_Den;
static const double delta2         = 0.5;       // Regularization factor in integrated resolution calculation
static double Wslope               = 5.;        // Slope with logE of weight of event for resolution contribution to utility.
static double Eslope               = 0.2;       // Slope of energy distribution for showers
static double DetectorSpacing      = 50.;       // Spacing of detector units. Defined in call argument list
static double SpacingStep          = 50.;       // Additional parameter used to define some of the detector geometries
static double Rslack               = 1000.;     // This parameter controls how far away to generate showers around array area
static int Ngrid                   = 100;       // Number of xy points for initial assay of likelihood 
static int NEgrid                  = 10;        // Number of energy values of initial search for shower likelihood
static double Einit[maxEinit];                  // Initial energy values for scan of likelihood in E
static double maxdTdR              = 10.;       // ns variation of arrival time per m variation of Reff
static double powbeta1[maxSteps];
static double powbeta2[maxSteps];
static double shift[10000];                     // array of random Gaussian shifts (for histogramming)
static double Ng_active;
static double Np_active;

// New random number generator
// ---------------------------
static TRandom3 * myRNG = new TRandom3();

// Shower parameters
// -----------------
static double PXeg1_p[3][3];
static double PXeg2_p[3][3];
static double PXeg3_p[3][3];
static double PXeg4_p[3][3];
static double PXmg1_p[3][3];
static double PXmg2_p[3][3];
static double PXmg3_p[3][3];
static double PXmg4_p[3][3];
static double PXep1_p[3][3];
static double PXep2_p[3][3];
static double PXep3_p[3][3];
static double PXep4_p[3][3];
static double PXmp1_p[3][3];
static double PXmp2_p[3][3];
static double PXmp3_p[3][3];
static double PXmp4_p[3][3];
// Parameters for lookup table of flux and derivatives:
static double thisp0_mg[100][100];
static double thisp1_mg[100][100];
static double thisp2_mg[100][100];
static double thisp0_eg[100][100];
static double thisp1_eg[100][100];
static double thisp2_eg[100][100];
static double thisp0_mp[100][100];
static double thisp1_mp[100][100];
static double thisp2_mp[100][100];
static double thisp0_ep[100][100];
static double thisp1_ep[100][100];
static double thisp2_ep[100][100];
static double dthisp0de_mg[100][100];
static double dthisp1de_mg[100][100];
static double dthisp2de_mg[100][100];
static double dthisp0de_eg[100][100];
static double dthisp1de_eg[100][100];
static double dthisp2de_eg[100][100];
static double dthisp0de_mp[100][100];
static double dthisp1de_mp[100][100];
static double dthisp2de_mp[100][100];
static double dthisp0de_ep[100][100];
static double dthisp1de_ep[100][100];
static double dthisp2de_ep[100][100];
static double dthisp0dth_mg[100][100];
static double dthisp1dth_mg[100][100];
static double dthisp2dth_mg[100][100];
static double dthisp0dth_eg[100][100];
static double dthisp1dth_eg[100][100];
static double dthisp2dth_eg[100][100];
static double dthisp0dth_mp[100][100];
static double dthisp1dth_mp[100][100];
static double dthisp2dth_mp[100][100];
static double dthisp0dth_ep[100][100];
static double dthisp1dth_ep[100][100];
static double dthisp2dth_ep[100][100];
static double detA;                             // For computation of cubic interpolation of pars
static double Ai[4][4];                         // same
static double A[4][4];                          // Matrix for solving cubics
static double B[4];                             // Initially contains values at four thetas, then converted in a,b,c,d of cubic
static double Y[4];                             // Values of cubic to interpolate fluxes
static double dBdY[4][4];                       // Derivatives of cubic pars vs Y
static double LLRmin; 
static double Utility;
static double UtilityErr;
static bool GammaHyp;
static double JS;                               // Jensen-Shannon divergence between the two densities of LLR

// Detector positions and parameters
// ---------------------------------
static double x[maxUnits];                      // Position of detectors in x
static double y[maxUnits];                      // Position of detectors in y 
static double xinit[maxUnits];                  // Initial position of detectors in x
static double yinit[maxUnits];                  // Initial position of detectors in y 
static double xprev[maxUnits];                  // Previous position in x 
static double yprev[maxUnits];                  // Previous position in y
static double TrueX0[maxEvents];                // X of center of generated shower on the ground
static double TrueY0[maxEvents];                // Y of center of generated shower on the ground
static double TrueTheta[maxEvents];             // Polar angle of shower
static double TruePhi[maxEvents];               // Azimuthal angle of shower. phi=0 along x direction.
static double TrueE[maxEvents];                 // Energy of shower in PeV
static bool Active[maxEvents];                  // Whether shower is considered (depends on number of tanks hit and P of likelihood fit)
static double PActive[maxEvents];               // Approximated probability that shower fires trigger, computed with exp values and Poisson approx
static double SumProbgt1[maxEvents];            // Needed for calculation of PActive above
static int Ntrigger;                            // Minimum number of tanks recording >0 particles for an event to be counted
static int F[maxNtrigger];                      // Lookup values of factorial, for sped up calcs
static int shape = 3;                           // 0 = hexagonal 1 = taxi 2 = spiral 3 = circular, and so on - this gets reassigned on start
static double spiral_reduction = 0.999;         // Geometric factor for spiral layout, see definelayout function, option 2
static double step_increase    = 1.02;          // Geometric factor for spiral layout, see definelayout function, option 2
static double dU_dxi[maxUnits];                 // Derivative of the utility vs x detector
static double dU_dyi[maxUnits];                 // Derivative of the utility vs y detector
static double sigma_time = 1.0;                 // Time resolution assumed for detectors, 1 ns
static double sigma_texp = sigma_time;          // Time resolution of expected arrival of secondaries, for now equal to the one above
static double sigma2_time = pow(sigma_time,2);  // Squared time resolution
static double sigma2_texp = sigma2_time;        
static double pg[maxEvents];
static double pp[maxEvents];
static double dIR_dEj[maxEvents];
static double LearningRateR[maxRbins];
static int NumAvgSteps;
static int DenAvgSteps;

// Number of mus and es detected in each detector unit
// ---------------------------------------------------
static float Nm[maxUnits][maxEvents];
static float Ne[maxUnits][maxEvents];
static float Tm[maxUnits][maxEvents];
static float Te[maxUnits][maxEvents];

// Measured values of position and angle of shower
// -----------------------------------------------
static double x0meas[maxEvents][2];             // Measured X0 of shower. There are two values, one for each hypothesis (gamma/p)
static double y0meas[maxEvents][2];             // Measured Y0 of shower. 
static double thmeas[maxEvents][2];             // Measured theta of shower.
static double phmeas[maxEvents][2];             // Measured phi of shower.
static double Emeas[maxEvents][2];              // Measured energy of shower.

// Test statistic discriminating gamma from proton showers, for current batch
// --------------------------------------------------------------------------
static double logLRT[maxEvents];                // For templates construction
static bool   IsGamma[maxEvents];               // Whether event is generated as a true gamma or proton shower
static double sigmaLRT[maxEvents];              // RMS of Log-likelihood ratio, needed for derivative calculation
static double dsigma2_dx[maxUnits][maxEvents];  // Derivative of LLR uncertainty squared of shower ik over effective distance x from detector i
static double dsigma2_dy[maxUnits][maxEvents];  // Derivative of LLR uncertainty squared of shower ik over effective distance y from detector i
static double Emin = 0.1;                       // In PeV. Lower boundary of energy in shower model
static double Emax = 10.;                       // Upper boundary
static double TrueGammaFraction;                // Fraction of gammas in generated batch
static double MeasFg;                           // Measured fraction of photons in batch from Likelihood using pg, pp PDF values
static double MeasFgErr;                        // Uncertainty on fraction as derived from fit to TS distribution
static double inv_sigmafs;
static double inv_sigmafs2;
static double sigmafs2;
static double LRE;                              // Learning rate on energy in shower reconstruction likelihood
static double LRX;                              // Learning rate on position in shower reconstruction likelihood
static double LRA;                              // Learning rate on angles in shower reconstruction likelihood
static const double MinLearningRate = 0.01;     // Min value of LR for SGD
static const double MaxLearningRate = 100.0;    // Max value of LR for SGD
static const double logLApprox      = 0.5;      // Determines precision of shower reconstruction likelihood (decreasing it slows down calcs)
static double ExposureFactor;
static int warnings1                = 0.;
static int warnings2                = 0.;
static int warnings3                = 0.;
static int warnings4                = 0.;
static int warnings5                = 0.;
static int warnings6                = 0.;
static ofstream outfile;

// Static histograms (ones that are handled in called functions)
// -------------------------------------------------------------
static double maxdxy    = 1000.;
#ifdef FEWPLOTS
static TProfile * DE    = new TProfile ("DE",    "Energy res. vs log(Etrue), running config", 9, log(0.01)/log_10, 1.001, 0., 1000.); 
static TProfile * DE0   = new TProfile ("DE0",   "Energy res. vs log(Etrue), initial config", 9, log(0.01)/log_10, 1.001, 0., 1000.); 
static TProfile * SvsSP = new TProfile ("SvsSP", "RMS of test statistic comparison", 100, 0., 10., 0., 20.);
static TH2D * SvsS      = new TH2D     ("SvsS",  "RMS of test statistic comparison", 100, 0., 10., 100, 0., 10.);
static TH2D * NumStepsvsxy;
static TH2D * NumStepsvsxyN; 
#endif
#ifdef PLOTS
static TH1D * DXP       = new TH1D     ("DXP",       "Difference between true and fit x for proton showers", 200, -maxdxy, maxdxy);
static TH1D * DYP       = new TH1D     ("DYP",       "Difference between true and fit y for proton showers", 200, -maxdxy, maxdxy);
static TH1D * DXG       = new TH1D     ("DXG",       "Difference between true and fit x for gamma showers",  200, -maxdxy, maxdxy);
static TH1D * DYG       = new TH1D     ("DYG",       "Difference between true and fit y for gamma showers",  200, -maxdxy, maxdxy);
static TH1D * DTHG      = new TH1D     ("DTHG",      "Difference between true and fit theta for gamma showers",  200, -halfpi, halfpi);
static TH1D * DPHG      = new TH1D     ("DPHG",      "Difference between true and fit phi for gamma showers",    200, -twopi, twopi);
static TH1D * DTHP      = new TH1D     ("DTHP",      "Difference between true and fit theta for proton showers", 200, -halfpi, halfpi);
static TH1D * DPHP      = new TH1D     ("DPHP",      "Difference between true and fit phi for proton showers",   200, -twopi, twopi);
static TH1D * DEG       = new TH1D     ("DEG",       "Difference between true and fit energy for gamma showers",   100, 0., 10.);
static TH1D * DEP       = new TH1D     ("DEP",       "Difference between true and fit energy for proton showers",  200, 0., 5.);
static TH2D * DTHPvsT   = new TH2D     ("DTHPvsT",   "Theta residual for proton showers vs theta", 50, -halfpi,halfpi,50, 0.,halfpi );
static TH2D * DTHGvsT   = new TH2D     ("DTHGvsT",   "Theta residual for gamma showers vs theta",  50, -halfpi,halfpi,50, 0.,halfpi );
static TH1D * DX0g      = new TH1D     ("DX0g",      "", 500, -100., 100.);
static TH1D * DY0g      = new TH1D     ("DY0g",      "", 500, -100., 100.);
static TH1D * DThg      = new TH1D     ("DThg",      "", 500, -halfpi, halfpi);
static TH1D * DPhg      = new TH1D     ("DPhg",      "", 500, 0., twopi);
static TH1D * DX0p      = new TH1D     ("DX0p",      "", 500, -100., 100.);
static TH1D * DY0p      = new TH1D     ("DY0p",      "", 500, -100., 100.);
static TH1D * DThp      = new TH1D     ("DThp",      "", 500, -halfpi, halfpi);
static TH1D * DPhp      = new TH1D     ("DPhp",      "", 500, 0., twopi);
static TH1D * SigLRT    = new TH1D     ("SigLRT",    "", 100, 0., 100000.);
static TH2D * SigLvsDRg = new TH2D     ("SigLvsDRg", "", 100, 0., 100000., 100, 0., 1000.);
static TH2D * SigLvsDRp = new TH2D     ("SigLvsDRp", "", 100, 0., 100000., 100, 0., 1000.);
static TH2D * DL        = new TH2D     ("DL",        "", 100, 0.5, 100.5,100,-50000.,+50000.);
static TH2D * NmuvsSh   = new TH2D     ("NmuvsSh",   "", 200, 0., 2000., 50, 0., 50.);
static TH2D * NevsSh    = new TH2D     ("NevsSh",    "", 200, 0., 2000., 50, 0., 50.);
static TProfile * NumStepsg; 
static TProfile * NumStepsp;  

//static TH2D * logLvsdr  = new TH2D     ("logLvsdr",  "", 200, -500., 0., 100, -10., 10.);
static TH2D * P0mg      = new TH2D     ("P0mg","", 100, 0., 100., 100, 0., 100.);
static TH2D * P2mg      = new TH2D     ("P2mg","", 100, 0., 100., 100, 0., 100.);
static TH2D * P0mp      = new TH2D     ("P0mp","", 100, 0., 100., 100, 0., 100.);
static TH2D * P2mp      = new TH2D     ("P2mp","", 100, 0., 100., 100, 0., 100.);
static TH2D * P0eg      = new TH2D     ("P0eg","", 100, 0., 100., 100, 0., 100.);
static TH2D * P1eg      = new TH2D     ("P1eg","", 100, 0., 100., 100, 0., 100.);
static TH2D * P2eg      = new TH2D     ("P2eg","", 100, 0., 100., 100, 0., 100.);
static TH2D * P0ep      = new TH2D     ("P0ep","", 100, 0., 100., 100, 0., 100.);
static TH2D * P1ep      = new TH2D     ("P1ep","", 100, 0., 100., 100, 0., 100.);
static TH2D * P2ep      = new TH2D     ("P2ep","", 100, 0., 100., 100, 0., 100.);
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Functions
// ---------

// Factorial function
// Note, it breaks down for N>170. Better use logfactorial below
// -------------------------------------------------------------
long double Factorial (int n) {
  if (n==0)  return 1.;
  if (n>170) { 
    cout << "     Warning - Factorial calculation breaks down for this n = " << n << endl;
    return largenumber;
  }
  return Factorial(n-1)*n;
}

// Log factorial function
// ----------------------
long double LogFactorial (int n) {
    // Use Stirling's approximation
    if (n==0) return 0.;
    return log(sqrt(twopi*n)) + (double)n*(log(n)-1.); 
}  

// Poisson function, with approximation for large numbers
// ------------------------------------------------------
double Poisson (int k, double mu) {
    double p;
    if (mu<0.) {
        warnings5++;
        return 0.;
    } else if (mu==0.) {
        return 0.;
    } else if (mu<15. && k<maxNtrigger) {
            p = exp(-mu)*pow(mu,k)/F[k];
    } else {
        // Use Gaussian approximation
        // --------------------------
        double s = sqrt(mu);
        p = exp(-0.5*pow((k-mu)/s,2.))/(sqrt2pi*s);
    }
    return p;
}

// Learning rate scheduler - this returns an oscillating, dampened function as a function of the epoch
// ---------------------------------------------------------------------------------------------------
double LR_Scheduler (int epoch) {
    double par[3] = {-0.02,0.8,0.2};
    double x = 100.*epoch/Nepochs;
    return exp(par[0]*x)*(par[1]+(1.-par[1])*pow(cos(par[2]*x),2));
}

// Derivation of the distance from the axis: we define the point on the axis of minimal distance
// to (x,y,0) as (a,b,c). Coordinates (x=a(t),y=b(t),z=c(t)) of points on the line fulfil the conditions
//    a = t * sin(theta) * cos(phi);
//    b = t * sin(theta) * sin(phi);
//    c = t * cos(theta). 
// [NB With this parametrization, positive t means negative z as phi is angle of positive propagation downward]
// We express the distance between the two points as
//    d^2 = (x-a)^2 + (y-b)^2 + c^2 = (x-t*st*cp)^2 + (y-t*st*sp)^2 + c^2
// This gets derived by t to obtain
//    dd^2/dt = -2*st*cp*(x-t*st*cp) -2*st*sp*(y-t*st*sp) +2*t * ct^2 = 0
// which can be solved for t to get
//    t = st*cp*x + st*sp*y
// Now we can compute d^2_min as
//    d^2_min = (x-t*st*cp)^2 + (y-t*st*sp)^2 + (t*ct)^2
// -------------------------------------------------------------------------------------------------------------
double EffectiveDistance (double xd, double yd, double x0, double y0, double theta, double phi, int mode) {
    double dx = xd-x0;
    double dy = yd-y0;
    double sp = sin(phi);
    double cp = cos(phi);
    double ct = cos(theta);
    double st = sin(theta);
    double r;
    // We treat separately the case of orthogonal showers, which is much easier and faster
    // -----------------------------------------------------------------------------------
    if (theta==0.) {
        r = dx*dx+dy*dy;
        if (r>0.) r = sqrt(r);
        if (r<Rmin) r = Rmin;
        if (mode==0) {
            return r;
        } else if (mode==1) { // derivative wrt x0
            return -dx/r;
        } else if (mode==2) { // derivative wrt y0
            return -dy/r;
        } else if (mode==3) { // derivative wrt theta 
            return 0.;
        } else if (mode==4) { // derivative wrt phi 
            return 0.;
        }
    } else {
        // NB below, the derivative wrt x0, y0 is very easy to compute using R^2 = dx^2 + dy^2 - t^2 
        // and remembering that t depends on x,y so r^2 = dx^2 + dy^2 -(dx*st*cp+dy*st*sp)^2
        // and we get the expressions reported below.
        // -----------------------------------------------------------------------------------------
        double stsp   = st*sp;
        double stcp   = st*cp;
        double t      = stcp*dx + stsp*dy;
        r             = dx*dx + dy*dy - t*t;
        if (r>0.)   r = sqrt(r);
        if (r<Rmin) r = Rmin;
        if (mode==0) {
            return r;
        } else if (mode==1) { // derivative wrt x0 (NNBB not x! Minus sign involved). 
            return -1./r * (dx-t*stcp);
        } else if (mode==2) { // derivative wrt y0 (NNBB not y! Minus sign involved)
            return -1./r * (dy-t*stsp);
        } else if (mode==3) { // derivative wrt theta
            // Derived using again R^2 = dx^2 + dy^2 - t^2, where t is solution of min distance
            // --------------------------------------------------------------------------------
            return -t*ct* (dx*cp + dy*sp) / r;
        } else if (mode==4) { // derivative wrt phi
            // Again simpler to use the above expression for r, getting:
            // ---------------------------------------------------------
            return -t*st * (-dx*sp + dy*cp) / r;
        }
    }
    cout << "     Something fishy in effective distance " << endl;
    return Rmin; // this should not happen
}

// The time of arrival on the ground depends on the variable t of the EffectiveDistance calculation above,
// and it has the correct sign (t>0 for later arrivals) with the definition of t there.
// -------------------------------------------------------------------------------------------------------
double EffectiveTime (double x, double y, double x0, double y0, double theta, double phi, int mode) {
    if (mode==0) {
        return ((x-x0)*sin(theta)*cos(phi) + (y-y0)*sin(theta)*sin(phi))/c0;
        // Note that we could obtain this, probably more economically, by doing T = sqrt(R^2-R_eff^2)
        // but then we'd still need to know the sign, which requires computing asin()
        // ------------------------------------------------------------------------------------------
    } else if (mode==1) { // derivative wrt x0
        return -sin(theta)*cos(phi)/c0; // note minus sign
    } else if (mode==2) { // derivative wrt y0
        return -sin(theta)*sin(phi)/c0; // note minus sign
    } else if (mode==3) { // derivative wrt theta
        return (cos(theta)*((x-x0)*cos(phi) + (y-y0)*sin(phi)))/c0;
    } else if (mode==4) { // derivative wrt phi
        return (sin(theta)*(-(x-x0)*sin(phi) + (y-y0)*cos(phi)))/c0;
    } 
    return 0.; 
}

// Inverse of 4x4 matrix necessary for interpolation of cubic
// ----------------------------------------------------------
void InitInverse4by4 () {
    //double a11 = A[0][0];
    //double a12 = A[0][1];
    //double a13 = A[0][2];
    //double a14 = A[0][3];
    //double a21 = A[1][0];
    //double a22 = A[1][1];
    //double a23 = A[1][2];
    //double a24 = A[1][3];
    //double a31 = A[2][0];
    //double a32 = A[2][1];
    //double a33 = A[2][2];
    //double a34 = A[2][3];
    //double a41 = A[3][0];
    //double a42 = A[3][1];
    //double a43 = A[3][2];
    //double a44 = A[3][3];
     double a11 = 1;
     double a12 = 1;
     double a13 = 1;
     double a14 = 1;
     double a21 = 1;
     double a22 = 2;
     double a23 = 4;
     double a24 = 8;
     double a31 = 1;
     double a32 = 3;
     double a33 = 9;
     double a34 = 27;
     double a41 = 1;
     double a42 = 4;
     double a43 = 16;
     double a44 = 64;


    // See https://semath.info/src/inverse-cofactor-ex4.html for the details of this explicit calculation
    // of the inverse of a 4x4 matrix. We use the explicit form to speed up calculations
    // --------------------------------------------------------------------------------------------------
    detA = a11*a22*a33*a44 + a11*a23*a34*a42 + a11*a24*a32*a43 -
           a11*a24*a33*a42 - a11*a23*a32*a44 - a11*a22*a34*a43 - 
           a12*a21*a33*a44 - a13*a21*a34*a42 - a14*a21*a32*a43 +
           a14*a21*a33*a42 + a13*a21*a32*a44 + a12*a21*a34*a43 +
           a12*a23*a31*a44 + a13*a24*a31*a42 + a14*a22*a31*a43 -
           a14*a23*a31*a42 - a13*a22*a31*a44 - a12*a24*a31*a43 -
           a12*a23*a34*a41 - a13*a24*a32*a41 - a14*a22*a33*a41 +
           a14*a23*a32*a41 + a13*a22*a34*a41 + a12*a24*a33*a41;

    Ai[0][0] =   a22*a33*a44 + a23*a34*a42 + a24*a32*a43 - 
                 a24*a33*a42 - a23*a32*a44 - a22*a34*a43;
    Ai[0][1] = - a12*a33*a44 - a13*a34*a42 - a14*a32*a43 +
                 a14*a33*a42 + a13*a32*a44 + a12*a34*a43;
    Ai[0][2] =   a12*a23*a44 + a13*a24*a42 + a14*a22*a43 -
                 a14*a23*a42 - a13*a22*a44 - a12*a24*a43;
    Ai[0][3] = - a12*a23*a34 - a13*a24*a32 - a14*a22*a33 +
                 a14*a23*a32 + a13*a22*a34 + a12*a24*a33;
    Ai[1][0] = - a21*a33*a44 - a23*a34*a41 - a24*a31*a43 +
                 a24*a33*a41 + a23*a31*a44 + a21*a34*a43;
    Ai[1][1] =   a11*a33*a44 + a13*a34*a41 + a14*a31*a43 -
                 a14*a33*a41 - a13*a31*a44 - a11*a34*a43;
    Ai[1][2] = - a11*a23*a44 - a13*a24*a41 - a14*a21*a43 +
                 a14*a23*a41 + a13*a21*a44 + a11*a24*a43;
    Ai[1][3] =   a11*a23*a34 + a13*a24*a31 + a14*a21*a33 -
                 a14*a23*a31 - a13*a21*a34 - a11*a24*a33;

    Ai[2][0] =   a21*a32*a44 + a22*a34*a41 + a24*a31*a42 -
                 a24*a32*a41 - a22*a31*a44 - a21*a34*a42;
    Ai[2][1] = - a11*a32*a44 - a12*a34*a41 - a14*a31*a42 +
                 a14*a32*a41 + a12*a31*a44 + a11*a34*a42;
    Ai[2][2] =   a11*a22*a44 + a12*a24*a41 + a14*a21*a42 -
                 a14*a22*a41 - a12*a21*a44 - a11*a24*a42;
    Ai[2][3] = - a11*a22*a34 - a12*a24*a31 - a14*a21*a32 +
                 a14*a22*a31 + a12*a21*a34 + a11*a24*a32;
                 
    Ai[3][0] = - a21*a32*a43 - a22*a33*a41 - a23*a31*a42 +
                 a23*a32*a41 + a22*a31*a43 + a21*a33*a42;
    Ai[3][1] =   a11*a32*a43 + a12*a33*a41 + a13*a31*a42 - 
                 a13*a32*a41 - a12*a31*a43 - a11*a33*a42;
    Ai[3][2] = - a11*a22*a43 - a12*a23*a41 - a13*a21*a42 +
                 a13*a22*a41 + a12*a21*a43 + a11*a23*a42;
    Ai[3][3] =   a11*a22*a33 + a12*a23*a31 + a13*a21*a32 -
                 a13*a22*a31 - a12*a21*a33 - a11*a23*a32;

    for (int i=0; i<4; i++) {
        for (int j=0; j<4; j++) {
            Ai[i][j] /= detA;
        }
    }
    return;
}

// Obtain the four parameters of a cubic passing through the four data points
// (1.,Y[0]), (2.,Y[1]), (3.,Y[2]), (4.,Y[3]) by multiplying the inverse matrix
// by the vector of known Y
// ----------------------------------------------------------------------------
void computecubicpars(int mode) {
    for (int i=0; i<4; i++) {
        B[i] = 0.;
    }
    if (mode==0) {
        for (int i=0; i<4; i++) {
            B[0] += Ai[0][i]*Y[i];
            B[1] += Ai[1][i]*Y[i];
            B[2] += Ai[2][i]*Y[i];
            B[3] += Ai[3][i]*Y[i];
        }
    } else if (mode==1) { // Derivatives with respect to Y[]. Here dBdY[i][j] is dB[i]/dY[j]
        for (int i=0; i<4; i++) {
            dBdY[0][i] = Ai[0][i];
            dBdY[1][i] = Ai[1][i];
            dBdY[2][i] = Ai[2][i];
            dBdY[3][i] = Ai[3][i];
        }
    } else {
        cout << "Invalid choice in computecubicpars." << endl;
    }
    return;
}               

// This function obtains the value of parameters thisp0, thisp2 for muons from gammas, or derivatives
// --------------------------------------------------------------------------------------------------
double solvecubic_mg (int parnumber, double energy, double theta, int mode) {

    if (mode==0) { // primal value
        computecubicpars (0); // <-- this will compute B[] given Y[]
        // Now B[] contains the four parameters of the cubic
        double val = 0.;
        double x = 0.5 + 4. * theta/thetamax;
        for (int i=0; i<4; i++) {
            val += B[i] * pow(x,i);
        }
        return val;
    } else if (mode==2) { // Get derivative wrt energy
        // In the above calculation, the value of the cubic at the four points
        // depends on energy through their parametrization, which is different
        // for the three parameters thisp0, thisp1, thisp2.
        // B are the parameters of the cubic that interpolates the Y points:
        //    thisp0 = B[0](Y[]) + B[1](Y[])*xt + B[2](Y[])*xt^2 + B[3](Y[])*xt^3
        // For thisp0, the dependence of the four values is
        //    Y[0] = exp(PXmg1[00]) + exp(PXmg1[01]*pow(f(e),PXmg1[02])) 
        //    Y[1] = exp(PXmg2[00]) + exp(PXmg2[01]*pow(f(e),PXmg2[02])) 
        //    Y[2] = exp(PXmg3[00]) + exp(PXmg3[01]*pow(f(e),PXmg3[02])) 
        //    Y[3] = exp(PXmg4[00]) + exp(PXmg4[01]*pow(f(e),PXmg4[02])) 
        // with 
        //    f(e)  = 0.5 + 20*(log(e)-log01)/(log10-log01)
        //    f'(e) = 20/(log10-log01)/e
        // So we get
        //    dB[0]/de = dBdY[0][0] * dY[0]/de + dBdY[0][1] * dY[1]/de + dBdY[0][2] * dY[2]/de + dBdY[0][3] * dY[3]/de =
        //             = dBdY[0][0] * PXmg1[01]*(PXmg1[02])*f(e)^(PXmg1[02]-1)*exp{}*f'(e) +
        //               dBdY[0][1] * PXmg2[01]*(PXmg2[02])*f(e)^(PXmg2[02]-1)*exp{}*f'(e) +
        //               dBdY[0][2] * PXmg3[01]*(PXmg3[02])*f(e)^(PXmg3[02]-1)*exp{}*f'(e) +   
        //               dBdY[0][3] * PXmg4[01]*(PXmg4[02])*f(e)^(PXmg4[02]-1)*exp{}*f'(e)   
        //    dB[1]/de = dBdY[1][0] * dY[0]/de + dBdY[1][1] * dY[1]/de + dBdY[1][2] * dY[2]/de + dBdY[1][3] * dY[3]/de
        //    dB[2]/de = dBdY[2][0] * dY[0]/de + dBdY[2][1] * dY[1]/de + dBdY[2][2] * dY[2]/de + dBdY[2][3] * dY[3]/de
        //    dB[3]/de = dBdY[3][0] * dY[0]/de + dBdY[3][1] * dY[1]/de + dBdY[3][2] * dY[2]/de + dBdY[3][3] * dY[3]/de
        // and finally
        //    dthisp0/de = dB[0]/de + dB[1]/de * xt + dB[2]/de * xt^2 + dB[3]/de * xt^3
        // ----------------------------------------------------------------------------
        // For thisp1, the dependence of the four values is
        //    thisp1 = B[0](Y[]) + B[1](Y[])*xt + B[2](Y[])*xt^2 + B[3](Y[])*xt^3
        //    Y[0] = PXmg1[10] + PXmg1[11] * f(e) + PXmg1[12] * f(e)^2 
        //    Y[1] = PXmg2[10] + PXmg2[11] * f(e) + PXmg2[12] * f(e)^2 
        //    Y[2] = PXmg3[10] + PXmg3[11] * f(e) + PXmg3[12] * f(e)^2 
        //    Y[3] = PXmg4[10] + PXmg4[11] * f(e) + PXmg4[12] * f(e)^2 
        // So we get
        //    dB[0]/de = dBdY[0][0] * dY[0]/de + dBdY[0][1] * dY[1]/de + dBdY[0][2] * dY[2]/de + dBdY[0][3] * dY[3]/de =
        //             = dBdY[0][0] * (PXmg1[11] * f'(e) + PXmg1[12] * 2*f(e)*f'(e)) +
        //               dBdY[0][1] * (PXmg2[11] * f'(e) + PXmg2[12] * 2*f(e)*f'(e)) +
        //               dBdY[0][2] * (PXmg3[11] * f'(e) + PXmg3[12] * 2*f(e)*f'(e)) +   
        //               dBdY[0][3] * (PXmg4[11] * f'(e) + PXmg4[12] * 2*f(e)*f'(e))    
        // and similarly
        //    dB[1]/de = dBdY[1][0] * dY[0]/de + dBdY[1][1] * dY[1]/de + dBdY[1][2] * dY[2]/de + dBdY[1][3] * dY[3]/de
        //    dB[2]/de = dBdY[2][0] * dY[0]/de + dBdY[2][1] * dY[1]/de + dBdY[2][2] * dY[2]/de + dBdY[2][3] * dY[3]/de
        //    dB[3]/de = dBdY[3][0] * dY[0]/de + dBdY[3][1] * dY[1]/de + dBdY[3][2] * dY[2]/de + dBdY[3][3] * dY[3]/de
        // and finally
        //    dthisp1/de = dB[0]/de + dB[1]/de * xt + dB[2]/de * xt^2 + dB[3]/de * xt^3

        // For thisp2, the dependence of the four values is
        //    thisp2 = B[0](Y[]) + B[1](Y[])*xt + B[2](Y[])*xt^2 + B[3](Y[])*xt^3
        //    Y[0] = PXmg1[20] + PXmg1[21] * f(e) + PXmg1[22] * f(e)^2 
        //    Y[1] = PXmg2[20] + PXmg2[21] * f(e) + PXmg2[22] * f(e)^2 
        //    Y[2] = PXmg3[20] + PXmg3[21] * f(e) + PXmg3[22] * f(e)^2 
        //    Y[3] = PXmg4[20] + PXmg4[21] * f(e) + PXmg4[22] * f(e)^2 
        // So we get
        //    dB[0]/de = dBdY[0][0] * dY[0]/de + dBdY[0][1] * dY[1]/de + dBdY[0][2] * dY[2]/de + dBdY[0][3] * dY[3]/de =
        //             = dBdY[0][0] * (PXmg1[21] * f'(e) + PXmg1[22] * 2*f(e)*f'(e)) +
        //               dBdY[0][1] * (PXmg2[21] * f'(e) + PXmg2[22] * 2*f(e)*f'(e)) +
        //               dBdY[0][2] * (PXmg3[21] * f'(e) + PXmg3[22] * 2*f(e)*f'(e)) +   
        //               dBdY[0][3] * (PXmg4[21] * f'(e) + PXmg4[22] * 2*f(e)*f'(e))    
        // and similarly
        //    dB[1]/de = dBdY[1][0] * dY[0]/de + dBdY[1][1] * dY[1]/de + dBdY[1][2] * dY[2]/de + dBdY[1][3] * dY[3]/de
        //    dB[2]/de = dBdY[2][0] * dY[0]/de + dBdY[2][1] * dY[1]/de + dBdY[2][2] * dY[2]/de + dBdY[2][3] * dY[3]/de
        //    dB[3]/de = dBdY[3][0] * dY[0]/de + dBdY[3][1] * dY[1]/de + dBdY[3][2] * dY[2]/de + dBdY[3][3] * dY[3]/de
        // and finally
        //    dthisp2/de = dB[0]/de + dB[1]/de * xt + dB[2]/de * xt^2 + dB[3]/de * xt^3

        // This is the calculation of the four cubic parameters dBdY[][]
        // -------------------------------------------------------------
        computecubicpars (1); // <-- this will compute B[] given Y[]

        // Now all ingredient for dthisp0/de, dthisp1/de, dthisp2/de are there
        // -------------------------------------------------------------------
        double f_e      = 0.5 + 20.*(log(energy)-log_01)/logdif;
        double fprime_e = 20./(log_10-log_01)/energy;
        double x        = 0.5 + 4. * theta/thetamax;
        double dB0de, dB1de, dB2de, dB3de;
        if (parnumber==0) {
            //    dthisp0/de = dB[0]/de + dB[1]/de * xt + dB[2]/de * xt^2 + dB[3]/de * xt^3
            //    dB[0]/de = dBdY[0][0] * dY[0]/de + dBdY[0][1] * dY[1]/de + dBdY[0][2] * dY[2]/de + dBdY[0][3] * dY[3]/de =
            //             = dBdY[0][0] * PXmg1[01]*(PXmg1[02]-1)*exp{}*f'(e) +
            //               dBdY[0][1] * PXmg2[01]*(PXmg2[02]-1)*exp{}*f'(e) +
            //               dBdY[0][2] * PXmg3[01]*(PXmg3[02]-1)*exp{}*f'(e) +   
            //               dBdY[0][3] * PXmg4[01]*(PXmg4[02]-1)*exp{}*f'(e)    
            double f0 = PXmg1_p[0][1]*PXmg1_p[0][2]*pow(f_e,PXmg1_p[0][2]-1.)*exp(PXmg1_p[0][1]*pow(f_e,PXmg1_p[0][2]))*fprime_e;
            double f1 = PXmg2_p[0][1]*PXmg2_p[0][2]*pow(f_e,PXmg2_p[0][2]-1.)*exp(PXmg2_p[0][1]*pow(f_e,PXmg2_p[0][2]))*fprime_e;
            double f2 = PXmg3_p[0][1]*PXmg3_p[0][2]*pow(f_e,PXmg3_p[0][2]-1.)*exp(PXmg3_p[0][1]*pow(f_e,PXmg3_p[0][2]))*fprime_e;
            double f3 = PXmg4_p[0][1]*PXmg4_p[0][2]*pow(f_e,PXmg4_p[0][2]-1.)*exp(PXmg4_p[0][1]*pow(f_e,PXmg4_p[0][2]))*fprime_e;
            dB0de = dBdY[0][0] * f0 +
                    dBdY[0][1] * f1 +
                    dBdY[0][2] * f2 +
                    dBdY[0][3] * f3;
            dB1de = dBdY[1][0] * f0 +
                    dBdY[1][1] * f1 +
                    dBdY[1][2] * f2 +
                    dBdY[1][3] * f3;
            dB2de = dBdY[2][0] * f0 +
                    dBdY[2][1] * f1 +
                    dBdY[2][2] * f2 +
                    dBdY[2][3] * f3;
            dB3de = dBdY[3][0] * f0 +
                    dBdY[3][1] * f1 +
                    dBdY[3][2] * f2 +
                    dBdY[3][3] * f3;
        } else if (parnumber==2) { // No parnumber 1 for muons
            //    dthisp2/de = dB[0]/de + dB[1]/de * xt + dB[2]/de * xt^2 + dB[3]/de * xt^3
            //    dB[0]/de = dBdY[0][0] * dY[0]/de + dBdY[0][1] * dY[1]/de + dBdY[0][2] * dY[2]/de + dBdY[0][3] * dY[3]/de =
            //             = dBdY[0][0] * (PXmg1[21] * f'(e) + PXmg1[22] * 2*f(e)*f'(e)) +
            //               dBdY[0][1] * (PXmg2[21] * f'(e) + PXmg2[22] * 2*f(e)*f'(e)) +
            //               dBdY[0][2] * (PXmg3[21] * f'(e) + PXmg3[22] * 2*f(e)*f'(e)) +   
            //               dBdY[0][3] * (PXmg4[21] * f'(e) + PXmg4[22] * 2*f(e)*f'(e))    
            double f1 = PXmg1_p[2][1]*fprime_e + PXmg1_p[2][2]*2.*f_e*fprime_e;
            double f2 = PXmg2_p[2][1]*fprime_e + PXmg2_p[2][2]*2.*f_e*fprime_e;
            double f3 = PXmg3_p[2][1]*fprime_e + PXmg3_p[2][2]*2.*f_e*fprime_e;
            double f4 = PXmg4_p[2][1]*fprime_e + PXmg4_p[2][2]*2.*f_e*fprime_e;
            dB0de = dBdY[0][0] * f1 + 
                    dBdY[0][1] * f2 + 
                    dBdY[0][2] * f3 + 
                    dBdY[0][3] * f4;
            dB1de = dBdY[1][0] * f1 + 
                    dBdY[1][1] * f2 + 
                    dBdY[1][2] * f3 + 
                    dBdY[1][3] * f4;
            dB2de = dBdY[2][0] * f1 + 
                    dBdY[2][1] * f2 + 
                    dBdY[2][2] * f3 + 
                    dBdY[2][3] * f4;
            dB3de = dBdY[3][0] * f1 + 
                    dBdY[3][1] * f2 + 
                    dBdY[3][2] * f3 + 
                    dBdY[3][3] * f4;
        }
        return dB0de + dB1de*x + dB2de*x*x + dB3de*x*x*x;
    } else if (mode==3) { // Derivative wrt theta
        computecubicpars (0);
        // Now B[] contains the four parameters of the cubic
        double x = 0.5 + 4. * theta/thetamax;
        double dxdtheta = 4./thetamax;
        return (B[1] + B[2]*2.*x + B[3] * 3.*x*x) * dxdtheta;
    } 
    // If it gets here an invalid value of mode was passed
    // ---------------------------------------------------
    cout << "     Warning - invalid mode for cubic" << endl;
    warnings1++;
    return 0.; // This should not happen
}

// This function obtains the value of parameters thisp0, thisp1, thisp2 for e+g from gammas, or derivatives
// --------------------------------------------------------------------------------------------------------
double solvecubic_eg (int parnumber, double energy, double theta, int mode) {

    if (mode==0) { // primal value

        computecubicpars (0); // <-- this will compute B[] given Y[]
        // Now B[] contains the four parameters of the cubic
        double val = 0.;
        double x = 0.5 + 4. * theta/thetamax;
        for (int i=0; i<4; i++) {
            val += B[i] * pow(x,i);
        }
        return val;
    } else if (mode==2) { // Get derivative wrt energy
        // In the above calculation, the value of the cubic at the four points
        // depends on energy through their parametrization, which is different
        // for the three parameters thisp0, thisp1, thisp2.
        //    thisp0 = B[0](Y[]) + B[1](Y[])*xt + B[2](Y[])*xt^2 + B[3](Y[])*xt^3
        // For thisp0, the dependence of the four values is
        //    Y[0] = PXeg1[00] * exp(PXeg1[01]*pow(f(e),PXeg1[02])) 
        //    Y[1] = PXeg2[00] * exp(PXeg2[01]*pow(f(e),PXeg2[02])) 
        //    Y[2] = PXeg3[00] * exp(PXeg3[01]*pow(f(e),PXeg3[02])) 
        //    Y[3] = PXeg4[00] * exp(PXeg4[01]*pow(f(e),PXeg4[02])) 
        // with 
        //    f(e)  = 0.5 + 20*(log(e)-log01)/(log10-log01)
        //    f'(e) = 20/(log10-log01)/e
        // So we get
        //    dB[0]/de = dBdY[0][0] * dY[0]/de + dBdY[0][1] * dY[1]/de + dBdY[0][2] * dY[2]/de + dBdY[0][3] * dY[3]/de =
        //             = dBdY[0][0] * PXeg1[00]*PXeg1[01]*PXeg1[02]*f_e^(PXeg1[02]-1)*exp{}*f'(e) +
        //               dBdY[0][1] * PXeg2[00]*PXeg2[01]*PXeg2[02]*f_e^(PXeg2[02]-1)*exp{}*f'(e) +
        //               dBdY[0][2] * PXeg3[00]*PXeg3[01]*PXeg3[02]*f_e^(PXeg3[02]-1)*exp{}*f'(e) +
        //               dBdY[0][3] * PXeg4[00]*PXeg4[01]*PXeg4[02]*f_e^(PXeg4[02]-1)*exp{}*f'(e) 
        //    dB[1]/de = dBdY[1][0] * dY[0]/de + dBdY[1][1] * dY[1]/de + dBdY[1][2] * dY[2]/de + dBdY[1][3] * dY[3]/de
        //    dB[2]/de = dBdY[2][0] * dY[0]/de + dBdY[2][1] * dY[1]/de + dBdY[2][2] * dY[2]/de + dBdY[2][3] * dY[3]/de
        //    dB[3]/de = dBdY[3][0] * dY[0]/de + dBdY[3][1] * dY[1]/de + dBdY[3][2] * dY[2]/de + dBdY[3][3] * dY[3]/de
        // and finally
        //    dthisp0/de = dB[0]/de + dB[1]/de * xt + dB[2]/de * xt^2 + dB[3]/de * xt^3
        // ----------------------------------------------------------------------------
        // For thisp1, the dependence of the four values is
        //    thisp1 = B[0](Y[]) + B[1](Y[])*xt + B[2](Y[])*xt^2 + B[3](Y[])*xt^3
        //    Y[0] = PXeg1[10] + PXeg1[11] * f(e) + PXeg1[12] * f(e)^2 
        //    Y[1] = PXeg2[10] + PXeg2[11] * f(e) + PXeg2[12] * f(e)^2 
        //    Y[2] = PXeg3[10] + PXeg3[11] * f(e) + PXeg3[12] * f(e)^2 
        //    Y[3] = PXeg4[10] + PXeg4[11] * f(e) + PXeg4[12] * f(e)^2 
        // So we get
        //    dB[0]/de = dBdY[0][0] * dY[0]/de + dBdY[0][1] * dY[1]/de + dBdY[0][2] * dY[2]/de + dBdY[0][3] * dY[3]/de =
        //             = dBdY[0][0] * (PXeg1[11] * f'(e) + PXeg1[12] * 2*f(e)*f'(e)) +
        //               dBdY[0][1] * (PXeg2[11] * f'(e) + PXeg2[12] * 2*f(e)*f'(e)) +
        //               dBdY[0][2] * (PXeg3[11] * f'(e) + PXeg3[12] * 2*f(e)*f'(e)) +   
        //               dBdY[0][3] * (PXeg4[11] * f'(e) + PXeg4[12] * 2*f(e)*f'(e))    
        // and similarly
        //    dB[1]/de = dBdY[1][0] * dY[0]/de + dBdY[1][1] * dY[1]/de + dBdY[1][2] * dY[2]/de + dBdY[1][3] * dY[3]/de
        //    dB[2]/de = dBdY[2][0] * dY[0]/de + dBdY[2][1] * dY[1]/de + dBdY[2][2] * dY[2]/de + dBdY[2][3] * dY[3]/de
        //    dB[3]/de = dBdY[3][0] * dY[0]/de + dBdY[3][1] * dY[1]/de + dBdY[3][2] * dY[2]/de + dBdY[3][3] * dY[3]/de
        // and finally
        //    dthisp1/de = dB[0]/de + dB[1]/de * xt + dB[2]/de * xt^2 + dB[3]/de * xt^3

        // For thisp2, the dependence of the four values is
        //    thisp2 = B[0](Y[]) + B[1](Y[])*xt + B[2](Y[])*xt^2 + B[3](Y[])*xt^3
        //    Y[0] = PXeg1[20] + PXeg1[21] * f(e) + PXeg1[22] * f(e)^2 
        //    Y[1] = PXeg2[20] + PXeg2[21] * f(e) + PXeg2[22] * f(e)^2 
        //    Y[2] = PXeg3[20] + PXeg3[21] * f(e) + PXeg3[22] * f(e)^2 
        //    Y[3] = PXeg4[20] + PXeg4[21] * f(e) + PXeg4[22] * f(e)^2 
        // So we get
        //    dB[0]/de = dBdY[0][0] * dY[0]/de + dBdY[0][1] * dY[1]/de + dBdY[0][2] * dY[2]/de + dBdY[0][3] * dY[3]/de =
        //             = dBdY[0][0] * (PXeg1[21] * f'(e) + PXeg1[22] * 2*f(e)*f'(e)) +
        //               dBdY[0][1] * (PXeg2[21] * f'(e) + PXeg2[22] * 2*f(e)*f'(e)) +
        //               dBdY[0][2] * (PXeg3[21] * f'(e) + PXeg3[22] * 2*f(e)*f'(e)) +   
        //               dBdY[0][3] * (PXeg4[21] * f'(e) + PXeg4[22] * 2*f(e)*f'(e))    
        // and similarly
        //    dB[1]/de = dBdY[1][0] * dY[0]/de + dBdY[1][1] * dY[1]/de + dBdY[1][2] * dY[2]/de + dBdY[1][3] * dY[3]/de
        //    dB[2]/de = dBdY[2][0] * dY[0]/de + dBdY[2][1] * dY[1]/de + dBdY[2][2] * dY[2]/de + dBdY[2][3] * dY[3]/de
        //    dB[3]/de = dBdY[3][0] * dY[0]/de + dBdY[3][1] * dY[1]/de + dBdY[3][2] * dY[2]/de + dBdY[3][3] * dY[3]/de
        // and finally
        //    dthisp2/de = dB[0]/de + dB[1]/de * xt + dB[2]/de * xt^2 + dB[3]/de * xt^3

        // This is the calculation of the four cubic parameters B[]
        computecubicpars (1); // <-- this will compute B[] given Y[]

        // Now all ingredient for dthisp0/de, dthisp1/de, dthisp2/de are there
        // -------------------------------------------------------------------
        double f_e      = 0.5 + 20.*(log(energy)-log_01)/logdif;
        double fprime_e = 20./(log_10-log_01)/energy;
        double x        = 0.5 + 4. * theta/thetamax;
        double dB0de, dB1de, dB2de, dB3de;
        if (parnumber==0) {
            //    dthisp0/de = dB[0]/de + dB[1]/de * xt + dB[2]/de * xt^2 + dB[3]/de * xt^3
            //    dB[0]/de = dBdY[0][0] * dY[0]/de + dBdY[0][1] * dY[1]/de + dBdY[0][2] * dY[2]/de + dBdY[0][3] * dY[3]/de =
            //             = dBdY[0][0] * PXeg1[00]*PXeg1[01]*(PXeg1[02]-1)*exp{}*f'(e) +
            //               dBdY[0][1] * PXeg2[00]*PXeg2[01]*(PXeg2[02]-1)*exp{}*f'(e) +
            //               dBdY[0][2] * PXeg3[00]*PXeg3[01]*(PXeg3[02]-1)*exp{}*f'(e) +   
            //               dBdY[0][3] * PXeg4[00]*PXeg4[01]*(PXeg4[02]-1)*exp{}*f'(e)    
            double f0 = PXeg1_p[0][0]*PXeg1_p[0][1]*PXeg1_p[0][2]*pow(f_e,PXeg1_p[0][2]-1.)*exp(PXeg1_p[0][1]*pow(f_e,PXeg1_p[0][2]))*fprime_e;
            double f1 = PXeg2_p[0][0]*PXeg2_p[0][1]*PXeg2_p[0][2]*pow(f_e,PXeg2_p[0][2]-1.)*exp(PXeg2_p[0][1]*pow(f_e,PXeg2_p[0][2]))*fprime_e;
            double f2 = PXeg3_p[0][0]*PXeg3_p[0][1]*PXeg3_p[0][2]*pow(f_e,PXeg3_p[0][2]-1.)*exp(PXeg3_p[0][1]*pow(f_e,PXeg3_p[0][2]))*fprime_e;
            double f3 = PXeg4_p[0][0]*PXeg4_p[0][1]*PXeg4_p[0][2]*pow(f_e,PXeg4_p[0][2]-1.)*exp(PXeg4_p[0][1]*pow(f_e,PXeg4_p[0][2]))*fprime_e;
            dB0de = dBdY[0][0] * f0 +
                    dBdY[0][1] * f1 +
                    dBdY[0][2] * f2 +
                    dBdY[0][3] * f3;
            dB1de = dBdY[1][0] * f0 +
                    dBdY[1][1] * f1 +
                    dBdY[1][2] * f2 +
                    dBdY[1][3] * f3;
            dB2de = dBdY[2][0] * f0 +
                    dBdY[2][1] * f1 +
                    dBdY[2][2] * f2 +
                    dBdY[2][3] * f3;
            dB3de = dBdY[3][0] * f0 +
                    dBdY[3][1] * f1 +
                    dBdY[3][2] * f2 +
                    dBdY[3][3] * f3;
        } else if (parnumber==1) {  
            //    dthisp2/de = dB[0]/de + dB[1]/de * xt + dB[2]/de * xt^2 + dB[3]/de * xt^3
            //    dB[0]/de = dBdY[0][0] * dY[0]/de + dBdY[0][1] * dY[1]/de + dBdY[0][2] * dY[2]/de + dBdY[0][3] * dY[3]/de =
            //             = dBdY[0][0] * (PXeg1[11] * f'(e) + PXeg1[12] * 2*f(e)*f'(e)) +
            //               dBdY[0][1] * (PXeg2[11] * f'(e) + PXeg2[12] * 2*f(e)*f'(e)) +
            //               dBdY[0][2] * (PXeg3[11] * f'(e) + PXeg3[12] * 2*f(e)*f'(e)) +   
            //               dBdY[0][3] * (PXeg4[11] * f'(e) + PXeg4[12] * 2*f(e)*f'(e))    
            double f1 = PXeg1_p[1][1]*fprime_e + PXeg1_p[1][2]*2.*f_e*fprime_e;
            double f2 = PXeg2_p[1][1]*fprime_e + PXeg2_p[1][2]*2.*f_e*fprime_e;
            double f3 = PXeg3_p[1][1]*fprime_e + PXeg3_p[1][2]*2.*f_e*fprime_e;
            double f4 = PXeg4_p[1][1]*fprime_e + PXeg4_p[1][2]*2.*f_e*fprime_e;
            dB0de = dBdY[0][0] * f1 + 
                    dBdY[0][1] * f2 + 
                    dBdY[0][2] * f3 + 
                    dBdY[0][3] * f4;
            dB1de = dBdY[1][0] * f1 + 
                    dBdY[1][1] * f2 + 
                    dBdY[1][2] * f3 + 
                    dBdY[1][3] * f4;
            dB2de = dBdY[2][0] * f1 + 
                    dBdY[2][1] * f2 + 
                    dBdY[2][2] * f3 + 
                    dBdY[2][3] * f4;
            dB3de = dBdY[3][0] * f1 + 
                    dBdY[3][1] * f2 + 
                    dBdY[3][2] * f3 + 
                    dBdY[3][3] * f4;
        } else if (parnumber==2) { 
            //    dthisp2/de = dB[0]/de + dB[1]/de * xt + dB[2]/de * xt^2 + dB[3]/de * xt^3
            //    dB[0]/de = dBdY[0][0] * dY[0]/de + dBdY[0][1] * dY[1]/de + dBdY[0][2] * dY[2]/de + dBdY[0][3] * dY[3]/de =
            //             = dBdY[0][0] * (PXeg1[21] * f'(e) + PXeg1[22] * 2*f(e)*f'(e)) +
            //               dBdY[0][1] * (PXeg2[21] * f'(e) + PXeg2[22] * 2*f(e)*f'(e)) +
            //               dBdY[0][2] * (PXeg3[21] * f'(e) + PXeg3[22] * 2*f(e)*f'(e)) +   
            //               dBdY[0][3] * (PXeg4[21] * f'(e) + PXeg4[22] * 2*f(e)*f'(e))    
            double f1 = PXeg1_p[2][1]*fprime_e + PXeg1_p[2][2]*2.*f_e*fprime_e;
            double f2 = PXeg2_p[2][1]*fprime_e + PXeg2_p[2][2]*2.*f_e*fprime_e;
            double f3 = PXeg3_p[2][1]*fprime_e + PXeg3_p[2][2]*2.*f_e*fprime_e;
            double f4 = PXeg4_p[2][1]*fprime_e + PXeg4_p[2][2]*2.*f_e*fprime_e;
            dB0de = dBdY[0][0] * f1 + 
                    dBdY[0][1] * f2 + 
                    dBdY[0][2] * f3 + 
                    dBdY[0][3] * f4;
            dB1de = dBdY[1][0] * f1 + 
                    dBdY[1][1] * f2 + 
                    dBdY[1][2] * f3 + 
                    dBdY[1][3] * f4;
            dB2de = dBdY[2][0] * f1 + 
                    dBdY[2][1] * f2 + 
                    dBdY[2][2] * f3 + 
                    dBdY[2][3] * f4;
            dB3de = dBdY[3][0] * f1 + 
                    dBdY[3][1] * f2 + 
                    dBdY[3][2] * f3 + 
                    dBdY[3][3] * f4;
        }
        return dB0de + dB1de*x + dB2de*x*x + dB3de*x*x*x;
    } else if (mode==3) { // Derivative wrt theta
        computecubicpars (0);
        // Now B[] contains the four parameters of the cubic
        double x = 0.5 + 4. * theta/thetamax;
        double dxdtheta = 4./thetamax;
        return (B[1] + B[2]*2.*x + B[3] * 3.*x*x) * dxdtheta;
    } 
    // If it gets here an invalid value of mode was passed
    // ---------------------------------------------------
    cout << "     Warning - invalid mode for cubic" << endl;
    warnings1++;
    return 0.; // This should not happen
}

// This function obtains the value of parameters thisp0, thisp2 for muons from protons, or derivatives
// ---------------------------------------------------------------------------------------------------
double solvecubic_mp (int parnumber, double energy, double theta, int mode) {

    if (mode==0) { // primal value
        computecubicpars (0); // <-- this will compute B[] given Y[]
        // Now B[] contains the four parameters of the cubic
        double val = 0.;
        double x = 0.5 + 4. * theta/thetamax;
        for (int i=0; i<4; i++) {
            val += B[i] * pow(x,i);
        }
        return val;
    } else if (mode==2) { // Get derivative wrt energy
        // In the above calculation, the value of the cubic at the four points
        // depends on energy through their parametrization, which is different
        // for the three parameters thisp0, thisp1, thisp2.
        //    thisp0 = B[0](Y[]) + B[1](Y[])*xt + B[2](Y[])*xt^2 + B[3](Y[])*xt^3
        // For thisp0, the dependence of the four values is
        //    Y[0] = exp(PXmp1[00]) + exp(PXmp1[01]*pow(f(e),PXmp1[02])) 
        //    Y[1] = exp(PXmp2[00]) + exp(PXmp2[01]*pow(f(e),PXmp2[02])) 
        //    Y[2] = exp(PXmp3[00]) + exp(PXmp3[01]*pow(f(e),PXmp3[02])) 
        //    Y[3] = exp(PXmp4[00]) + exp(PXmp4[01]*pow(f(e),PXmp4[02])) 
        // with 
        //    f(e)  = 0.5 + 20*(log(e)-log01)/(log10-log01)
        //    f'(e) = 20/(log10-log01)/e
        // So we get
        //    dB[0]/de = dBdY[0][0] * dY[0]/de + dBdY[0][1] * dY[1]/de + dBdY[0][2] * dY[2]/de + dBdY[0][3] * dY[3]/de =
        //             = dBdY[0][0] * PXmp1[01]*PXmp1[02]*f_e^(PXmp1[02]-1)*exp{}*f'(e) +
        //               dBdY[0][1] * PXmp2[01]*PXmp2[02]*f_e^(PXmp2[02]-1)*exp{}*f'(e) +
        //               dBdY[0][2] * PXmp3[01]*PXmp3[02]*f_e^(PXmp3[02]-1)*exp{}*f'(e) +   
        //               dBdY[0][3] * PXmp4[01]*PXmp4[02]*f_e^(PXmp4[02]-1)*exp{}*f'(e)    
        //    dB[1]/de = dBdY[1][0] * dY[0]/de + dBdY[1][1] * dY[1]/de + dBdY[1][2] * dY[2]/de + dBdY[1][3] * dY[3]/de
        //    dB[2]/de = dBdY[2][0] * dY[0]/de + dBdY[2][1] * dY[1]/de + dBdY[2][2] * dY[2]/de + dBdY[2][3] * dY[3]/de
        //    dB[3]/de = dBdY[3][0] * dY[0]/de + dBdY[3][1] * dY[1]/de + dBdY[3][2] * dY[2]/de + dBdY[3][3] * dY[3]/de
        // and finally
        //    dthisp0/de = dB[0]/de + dB[1]/de * xt + dB[2]/de * xt^2 + dB[3]/de * xt^3
        // ----------------------------------------------------------------------------
        // For thisp1, the dependence of the four values is
        //    thisp1 = B[0](Y[]) + B[1](Y[])*xt + B[2](Y[])*xt^2 + B[3](Y[])*xt^3
        //    Y[0] = PXmp1[10] + PXmp1[11] * f(e) + PXmp1[12] * f(e)^2 
        //    Y[1] = PXmp2[10] + PXmp2[11] * f(e) + PXmp2[12] * f(e)^2 
        //    Y[2] = PXmp3[10] + PXmp3[11] * f(e) + PXmp3[12] * f(e)^2 
        //    Y[3] = PXmp4[10] + PXmp4[11] * f(e) + PXmp4[12] * f(e)^2 
        // So we get
        //    dB[0]/de = dBdY[0][0] * dY[0]/de + dBdY[0][1] * dY[1]/de + dBdY[0][2] * dY[2]/de + dBdY[0][3] * dY[3]/de =
        //             = dBdY[0][0] * (PXmp1[11] * f'(e) + PXmp1[12] * 2*f(e)*f'(e)) +
        //               dBdY[0][1] * (PXmp2[11] * f'(e) + PXmp2[12] * 2*f(e)*f'(e)) +
        //               dBdY[0][2] * (PXmp3[11] * f'(e) + PXmp3[12] * 2*f(e)*f'(e)) +   
        //               dBdY[0][3] * (PXmp4[11] * f'(e) + PXmp4[12] * 2*f(e)*f'(e))    
        // and similarly
        //    dB[1]/de = dBdY[1][0] * dY[0]/de + dBdY[1][1] * dY[1]/de + dBdY[1][2] * dY[2]/de + dBdY[1][3] * dY[3]/de
        //    dB[2]/de = dBdY[2][0] * dY[0]/de + dBdY[2][1] * dY[1]/de + dBdY[2][2] * dY[2]/de + dBdY[2][3] * dY[3]/de
        //    dB[3]/de = dBdY[3][0] * dY[0]/de + dBdY[3][1] * dY[1]/de + dBdY[3][2] * dY[2]/de + dBdY[3][3] * dY[3]/de
        // and finally
        //    dthisp1/de = dB[0]/de + dB[1]/de * xt + dB[2]/de * xt^2 + dB[3]/de * xt^3

        // For thisp2, the dependence of the four values is
        //    thisp2 = B[0](Y[]) + B[1](Y[])*xt + B[2](Y[])*xt^2 + B[3](Y[])*xt^3
        //    Y[0] = PXmp1[20] + PXmp1[21] * f(e) + PXmp1[22] * f(e)^2 
        //    Y[1] = PXmp2[20] + PXmp2[21] * f(e) + PXmp2[22] * f(e)^2 
        //    Y[2] = PXmp3[20] + PXmp3[21] * f(e) + PXmp3[22] * f(e)^2 
        //    Y[3] = PXmp4[20] + PXmp4[21] * f(e) + PXmp4[22] * f(e)^2 
        // So we get
        //    dB[0]/de = dBdY[0][0] * dY[0]/de + dBdY[0][1] * dY[1]/de + dBdY[0][2] * dY[2]/de + dBdY[0][3] * dY[3]/de =
        //             = dBdY[0][0] * (PXmp1[21] * f'(e) + PXmp1[22] * 2*f(e)*f'(e)) +
        //               dBdY[0][1] * (PXmp2[21] * f'(e) + PXmp2[22] * 2*f(e)*f'(e)) +
        //               dBdY[0][2] * (PXmp3[21] * f'(e) + PXmp3[22] * 2*f(e)*f'(e)) +   
        //               dBdY[0][3] * (PXmp4[21] * f'(e) + PXmp4[22] * 2*f(e)*f'(e))    
        // and similarly
        //    dB[1]/de = dBdY[1][0] * dY[0]/de + dBdY[1][1] * dY[1]/de + dBdY[1][2] * dY[2]/de + dBdY[1][3] * dY[3]/de
        //    dB[2]/de = dBdY[2][0] * dY[0]/de + dBdY[2][1] * dY[1]/de + dBdY[2][2] * dY[2]/de + dBdY[2][3] * dY[3]/de
        //    dB[3]/de = dBdY[3][0] * dY[0]/de + dBdY[3][1] * dY[1]/de + dBdY[3][2] * dY[2]/de + dBdY[3][3] * dY[3]/de
        // and finally
        //    dthisp2/de = dB[0]/de + dB[1]/de * xt + dB[2]/de * xt^2 + dB[3]/de * xt^3

        // This is the calculation of the four cubic parameters B[]
        computecubicpars (1); // <-- this will compute B[] given Y[]

        // Now all ingredient for dthisp0/de, dthisp1/de, dthisp2/de are there
        // -------------------------------------------------------------------
        double f_e      = 0.5 + 20.*(log(energy)-log_01)/logdif;
        double fprime_e = 20./(log_10-log_01)/energy;
        double x        = 0.5 + 4. * theta/thetamax;
        double dB0de, dB1de, dB2de, dB3de;
        if (parnumber==0) {
        //    dthisp0/de = dB[0]/de + dB[1]/de * xt + dB[2]/de * xt^2 + dB[3]/de * xt^3
        //    dB[0]/de = dBdY[0][0] * dY[0]/de + dBdY[0][1] * dY[1]/de + dBdY[0][2] * dY[2]/de + dBdY[0][3] * dY[3]/de =
        //             = dBdY[0][0] * PXmp1[01]*(PXmp1[02]-1)*exp{}*f'(e) +
        //               dBdY[0][1] * PXmp2[01]*(PXmp2[02]-1)*exp{}*f'(e) +
        //               dBdY[0][2] * PXmp3[01]*(PXmp3[02]-1)*exp{}*f'(e) +   
        //               dBdY[0][3] * PXmp4[01]*(PXmp4[02]-1)*exp{}*f'(e)    
            double f0 = PXmp1_p[0][1]*PXmp1_p[0][2]*pow(f_e,PXmp1_p[0][2]-1.)*exp(PXmp1_p[0][1]*pow(f_e,PXmp1_p[0][2]))*fprime_e;
            double f1 = PXmp2_p[0][1]*PXmp2_p[0][2]*pow(f_e,PXmp2_p[0][2]-1.)*exp(PXmp2_p[0][1]*pow(f_e,PXmp2_p[0][2]))*fprime_e;
            double f2 = PXmp3_p[0][1]*PXmp3_p[0][2]*pow(f_e,PXmp3_p[0][2]-1.)*exp(PXmp3_p[0][1]*pow(f_e,PXmp3_p[0][2]))*fprime_e;
            double f3 = PXmp4_p[0][1]*PXmp4_p[0][2]*pow(f_e,PXmp4_p[0][2]-1.)*exp(PXmp4_p[0][1]*pow(f_e,PXmp4_p[0][2]))*fprime_e;
            dB0de = dBdY[0][0] * f0 +
                    dBdY[0][1] * f1 +
                    dBdY[0][2] * f2 +
                    dBdY[0][3] * f3;
            dB1de = dBdY[1][0] * f0 +
                    dBdY[1][1] * f1 +
                    dBdY[1][2] * f2 +
                    dBdY[1][3] * f3;
            dB2de = dBdY[2][0] * f0 +
                    dBdY[2][1] * f1 +
                    dBdY[2][2] * f2 +
                    dBdY[2][3] * f3;
            dB3de = dBdY[3][0] * f0 +
                    dBdY[3][1] * f1 +
                    dBdY[3][2] * f2 +
                    dBdY[3][3] * f3;
        } else if (parnumber==2) { // No parnumber 1 for muons
            //    dthisp2/de = dB[0]/de + dB[1]/de * xt + dB[2]/de * xt^2 + dB[3]/de * xt^3
            //    dB[0]/de = dBdY[0][0] * dY[0]/de + dBdY[0][1] * dY[1]/de + dBdY[0][2] * dY[2]/de + dBdY[0][3] * dY[3]/de =
            //             = dBdY[0][0] * (PXmp1[21] * f'(e) + PXmp1[22] * 2*f(e)*f'(e)) +
            //               dBdY[0][1] * (PXmp2[21] * f'(e) + PXmp2[22] * 2*f(e)*f'(e)) +
            //               dBdY[0][2] * (PXmp3[21] * f'(e) + PXmp3[22] * 2*f(e)*f'(e)) +   
            //               dBdY[0][3] * (PXmp4[21] * f'(e) + PXmp4[22] * 2*f(e)*f'(e))    
            double f1 = PXmp1_p[2][1]*fprime_e + PXmp1_p[2][2]*2.*f_e*fprime_e;
            double f2 = PXmp2_p[2][1]*fprime_e + PXmp2_p[2][2]*2.*f_e*fprime_e;
            double f3 = PXmp3_p[2][1]*fprime_e + PXmp3_p[2][2]*2.*f_e*fprime_e;
            double f4 = PXmp4_p[2][1]*fprime_e + PXmp4_p[2][2]*2.*f_e*fprime_e;
            dB0de = dBdY[0][0] * f1 + 
                    dBdY[0][1] * f2 + 
                    dBdY[0][2] * f3 + 
                    dBdY[0][3] * f4;
            dB1de = dBdY[1][0] * f1 + 
                    dBdY[1][1] * f2 + 
                    dBdY[1][2] * f3 + 
                    dBdY[1][3] * f4;
            dB2de = dBdY[2][0] * f1 + 
                    dBdY[2][1] * f2 + 
                    dBdY[2][2] * f3 + 
                    dBdY[2][3] * f4;
            dB3de = dBdY[3][0] * f1 + 
                    dBdY[3][1] * f2 + 
                    dBdY[3][2] * f3 + 
                    dBdY[3][3] * f4;
        }
        return dB0de + dB1de*x + dB2de*x*x + dB3de*x*x*x;
    } else if (mode==3) { // Derivative wrt theta
        computecubicpars (0);
        // Now B[] contains the four parameters of the cubic
        double x = 0.5 + 4. * theta/thetamax;
        double dxdtheta = 4./thetamax;
        return (B[1] + B[2]*2.*x + B[3] * 3.*x*x) * dxdtheta;
    } 
    // If it gets here an invalid value of mode was passed
    // ---------------------------------------------------
    cout << "     Warning - invalid mode for cubic" << endl;
    warnings1++;
    return 0.; // This should not happen
}

// This function obtains the value of parameters thisp0, thisp1, thisp2 for e+g from protons, or derivatives
// ---------------------------------------------------------------------------------------------------------
double solvecubic_ep (int parnumber, double energy, double theta, int mode) {

    if (mode==0) { // primal value
        computecubicpars (0); // <-- this will compute B[] given Y[]
        // We proceed to compute the four cubic parameters 
        // Now B[] contains the four parameters of the cubic
        double val = 0.;
        double x = 0.5 + 4. * theta/thetamax;
        for (int i=0; i<4; i++) {
            val += B[i] * pow(x,i);
        }
        return val;
    } else if (mode==2) { // Get derivative wrt energy
        // In the above calculation, the value of the cubic at the four points
        // depends on energy through their parametrization, which is different
        // for the three parameters thisp0, thisp1, thisp2.
        //    thisp0 = B[0](Y[]) + B[1](Y[])*xt + B[2](Y[])*xt^2 + B[3](Y[])*xt^3
        // For thisp0, the dependence of the four values is
        //    Y[0] = exp(PXep1[00]) + exp(PXep1[01]*pow(f(e),PXep1[02])) 
        //    Y[1] = exp(PXep2[00]) + exp(PXep2[01]*pow(f(e),PXep2[02])) 
        //    Y[2] = exp(PXep3[00]) + exp(PXep3[01]*pow(f(e),PXep3[02])) 
        //    Y[3] = exp(PXep4[00]) + exp(PXep4[01]*pow(f(e),PXep4[02])) 
        // with 
        //    f(e)  = 0.5 + 20*(log(e)-log01)/(log10-log01)
        //    f'(e) = 20/(log10-log01)/e
        // So we get
        //    dB[0]/de = dBdY[0][0] * dY[0]/de + dBdY[0][1] * dY[1]/de + dBdY[0][2] * dY[2]/de + dBdY[0][3] * dY[3]/de =
        //             = dBdY[0][0] * PXep1[01]*PXep1[02]*f_e^(PXep1[02]-1)*exp{}*f'(e) +
        //               dBdY[0][1] * PXep2[01]*PXep2[02]*f_e^(PXep2[02]-1)*exp{}*f'(e) +
        //               dBdY[0][2] * PXep3[01]*PXep3[02]*f_e^(PXep3[02]-1)*exp{}*f'(e) +   
        //               dBdY[0][3] * PXep4[01]*PXep4[02]*f_e^(PXep4[02]-1)*exp{}*f'(e)    
        //    dB[1]/de = dBdY[1][0] * dY[0]/de + dBdY[1][1] * dY[1]/de + dBdY[1][2] * dY[2]/de + dBdY[1][3] * dY[3]/de
        //    dB[2]/de = dBdY[2][0] * dY[0]/de + dBdY[2][1] * dY[1]/de + dBdY[2][2] * dY[2]/de + dBdY[2][3] * dY[3]/de
        //    dB[3]/de = dBdY[3][0] * dY[0]/de + dBdY[3][1] * dY[1]/de + dBdY[3][2] * dY[2]/de + dBdY[3][3] * dY[3]/de
        // and finally
        //    dthisp0/de = dB[0]/de + dB[1]/de * xt + dB[2]/de * xt^2 + dB[3]/de * xt^3
        // ----------------------------------------------------------------------------
        // For thisp1, the dependence of the four values is
        //    thisp1 = B[0](Y[]) + B[1](Y[])*xt + B[2](Y[])*xt^2 + B[3](Y[])*xt^3
        //    Y[0] = PXep1[10] + PXep1[11] * f(e) + PXep1[12] * f(e)^2 
        //    Y[1] = PXep2[10] + PXep2[11] * f(e) + PXep2[12] * f(e)^2 
        //    Y[2] = PXep3[10] + PXep3[11] * f(e) + PXep3[12] * f(e)^2 
        //    Y[3] = PXep4[10] + PXep4[11] * f(e) + PXep4[12] * f(e)^2 
        // So we get
        //    dB[0]/de = dBdY[0][0] * dY[0]/de + dBdY[0][1] * dY[1]/de + dBdY[0][2] * dY[2]/de + dBdY[0][3] * dY[3]/de =
        //             = dBdY[0][0] * (PXep1[11] * f'(e) + PXep1[12] * 2*f(e)*f'(e)) +
        //               dBdY[0][1] * (PXep2[11] * f'(e) + PXep2[12] * 2*f(e)*f'(e)) +
        //               dBdY[0][2] * (PXep3[11] * f'(e) + PXep3[12] * 2*f(e)*f'(e)) +   
        //               dBdY[0][3] * (PXep4[11] * f'(e) + PXep4[12] * 2*f(e)*f'(e))    
        // and similarly
        //    dB[1]/de = dBdY[1][0] * dY[0]/de + dBdY[1][1] * dY[1]/de + dBdY[1][2] * dY[2]/de + dBdY[1][3] * dY[3]/de
        //    dB[2]/de = dBdY[2][0] * dY[0]/de + dBdY[2][1] * dY[1]/de + dBdY[2][2] * dY[2]/de + dBdY[2][3] * dY[3]/de
        //    dB[3]/de = dBdY[3][0] * dY[0]/de + dBdY[3][1] * dY[1]/de + dBdY[3][2] * dY[2]/de + dBdY[3][3] * dY[3]/de
        // and finally
        //    dthisp1/de = dB[0]/de + dB[1]/de * xt + dB[2]/de * xt^2 + dB[3]/de * xt^3

        // For thisp2, the dependence of the four values is
        //    thisp2 = B[0](Y[]) + B[1](Y[])*xt + B[2](Y[])*xt^2 + B[3](Y[])*xt^3
        //    Y[0] = PXep1[20] + PXep1[21] * f(e) + PXep1[22] * f(e)^2 
        //    Y[1] = PXep2[20] + PXep2[21] * f(e) + PXep2[22] * f(e)^2 
        //    Y[2] = PXep3[20] + PXep3[21] * f(e) + PXep3[22] * f(e)^2 
        //    Y[3] = PXep4[20] + PXep4[21] * f(e) + PXep4[22] * f(e)^2 
        // So we get
        //    dB[0]/de = dBdY[0][0] * dY[0]/de + dBdY[0][1] * dY[1]/de + dBdY[0][2] * dY[2]/de + dBdY[0][3] * dY[3]/de =
        //             = dBdY[0][0] * (PXep1[21] * f'(e) + PXep1[22] * 2*f(e)*f'(e)) +
        //               dBdY[0][1] * (PXep2[21] * f'(e) + PXep2[22] * 2*f(e)*f'(e)) +
        //               dBdY[0][2] * (PXep3[21] * f'(e) + PXep3[22] * 2*f(e)*f'(e)) +   
        //               dBdY[0][3] * (PXep4[21] * f'(e) + PXep4[22] * 2*f(e)*f'(e))    
        // and similarly
        //    dB[1]/de = dBdY[1][0] * dY[0]/de + dBdY[1][1] * dY[1]/de + dBdY[1][2] * dY[2]/de + dBdY[1][3] * dY[3]/de
        //    dB[2]/de = dBdY[2][0] * dY[0]/de + dBdY[2][1] * dY[1]/de + dBdY[2][2] * dY[2]/de + dBdY[2][3] * dY[3]/de
        //    dB[3]/de = dBdY[3][0] * dY[0]/de + dBdY[3][1] * dY[1]/de + dBdY[3][2] * dY[2]/de + dBdY[3][3] * dY[3]/de
        // and finally
        //    dthisp2/de = dB[0]/de + dB[1]/de * xt + dB[2]/de * xt^2 + dB[3]/de * xt^3

        // This is the calculation of the four cubic parameters B[]
        computecubicpars (1); // <-- this will compute B[] given Y[]

        // Now all ingredient for dthisp0/de, dthisp1/de, dthisp2/de are there
        // -------------------------------------------------------------------
        double f_e      = 0.5 + 20.*(log(energy)-log_01)/logdif;
        double fprime_e = 20./(log_10-log_01)/energy;
        double x        = 0.5 + 4. * theta/thetamax;
        double dB0de, dB1de, dB2de, dB3de;
        if (parnumber==0) {
            //    dthisp0/de = dB[0]/de + dB[1]/de * xt + dB[2]/de * xt^2 + dB[3]/de * xt^3
            //    dB[0]/de = dBdY[0][0] * dY[0]/de + dBdY[0][1] * dY[1]/de + dBdY[0][2] * dY[2]/de + dBdY[0][3] * dY[3]/de =
            //             = dBdY[0][0] * PXep1[01]*(PXep1[02]-1)*exp{}*f'(e) +
            //               dBdY[0][1] * PXep2[01]*(PXep2[02]-1)*exp{}*f'(e) +
            //               dBdY[0][2] * PXep3[01]*(PXep3[02]-1)*exp{}*f'(e) +   
            //               dBdY[0][3] * PXep4[01]*(PXep4[02]-1)*exp{}*f'(e)    
            double f0 = PXep1_p[0][1]*PXep1_p[0][2]*pow(f_e,PXep1_p[0][2]-1.)*exp(PXep1_p[0][1]*pow(f_e,PXep1_p[0][2]))*fprime_e;
            double f1 = PXep2_p[0][1]*PXep2_p[0][2]*pow(f_e,PXep2_p[0][2]-1.)*exp(PXep2_p[0][1]*pow(f_e,PXep2_p[0][2]))*fprime_e;
            double f2 = PXep3_p[0][1]*PXep3_p[0][2]*pow(f_e,PXep3_p[0][2]-1.)*exp(PXep3_p[0][1]*pow(f_e,PXep3_p[0][2]))*fprime_e;
            double f3 = PXep4_p[0][1]*PXep4_p[0][2]*pow(f_e,PXep4_p[0][2]-1.)*exp(PXep4_p[0][1]*pow(f_e,PXep4_p[0][2]))*fprime_e;
            dB0de = dBdY[0][0] * f0 +
                    dBdY[0][1] * f1 +
                    dBdY[0][2] * f2 +
                    dBdY[0][3] * f3;
            dB1de = dBdY[1][0] * f0 +
                    dBdY[1][1] * f1 +
                    dBdY[1][2] * f2 +
                    dBdY[1][3] * f3;
            dB2de = dBdY[2][0] * f0 +
                    dBdY[2][1] * f1 +
                    dBdY[2][2] * f2 +
                    dBdY[2][3] * f3;
            dB3de = dBdY[3][0] * f0 +
                    dBdY[3][1] * f1 +
                    dBdY[3][2] * f2 +
                    dBdY[3][3] * f3;
        } else if (parnumber==1) {  
            //    dthisp2/de = dB[0]/de + dB[1]/de * xt + dB[2]/de * xt^2 + dB[3]/de * xt^3
            //    dB[0]/de = dBdY[0][0] * dY[0]/de + dBdY[0][1] * dY[1]/de + dBdY[0][2] * dY[2]/de + dBdY[0][3] * dY[3]/de =
            //             = dBdY[0][0] * (PXep1[11] * f'(e) + PXep1[12] * 2*f(e)*f'(e)) +
            //               dBdY[0][1] * (PXep2[11] * f'(e) + PXep2[12] * 2*f(e)*f'(e)) +
            //               dBdY[0][2] * (PXep3[11] * f'(e) + PXep3[12] * 2*f(e)*f'(e)) +   
            //               dBdY[0][3] * (PXep4[11] * f'(e) + PXep4[12] * 2*f(e)*f'(e))    
            double f1 = PXep1_p[1][1]*fprime_e + PXep1_p[1][2]*2.*f_e*fprime_e;
            double f2 = PXep2_p[1][1]*fprime_e + PXep2_p[1][2]*2.*f_e*fprime_e;
            double f3 = PXep3_p[1][1]*fprime_e + PXep3_p[1][2]*2.*f_e*fprime_e;
            double f4 = PXep4_p[1][1]*fprime_e + PXep4_p[1][2]*2.*f_e*fprime_e;
            dB0de = dBdY[0][0] * f1 + 
                    dBdY[0][1] * f2 + 
                    dBdY[0][2] * f3 + 
                    dBdY[0][3] * f4;
            dB1de = dBdY[1][0] * f1 + 
                    dBdY[1][1] * f2 + 
                    dBdY[1][2] * f3 + 
                    dBdY[1][3] * f4;
            dB2de = dBdY[2][0] * f1 + 
                    dBdY[2][1] * f2 + 
                    dBdY[2][2] * f3 + 
                    dBdY[2][3] * f4;
            dB3de = dBdY[3][0] * f1 + 
                    dBdY[3][1] * f2 + 
                    dBdY[3][2] * f3 + 
                    dBdY[3][3] * f4;
        } else if (parnumber==2) { 
            //    dthisp2/de = dB[0]/de + dB[1]/de * xt + dB[2]/de * xt^2 + dB[3]/de * xt^3
            //    dB[0]/de = dBdY[0][0] * dY[0]/de + dBdY[0][1] * dY[1]/de + dBdY[0][2] * dY[2]/de + dBdY[0][3] * dY[3]/de =
            //             = dBdY[0][0] * (PXep1[21] * f'(e) + PXep1[22] * 2*f(e)*f'(e)) +
            //               dBdY[0][1] * (PXep2[21] * f'(e) + PXep2[22] * 2*f(e)*f'(e)) +
            //               dBdY[0][2] * (PXep3[21] * f'(e) + PXep3[22] * 2*f(e)*f'(e)) +   
            //               dBdY[0][3] * (PXep4[21] * f'(e) + PXep4[22] * 2*f(e)*f'(e))    
            double f1 = PXep1_p[2][1]*fprime_e + PXep1_p[2][2]*2.*f_e*fprime_e;
            double f2 = PXep2_p[2][1]*fprime_e + PXep2_p[2][2]*2.*f_e*fprime_e;
            double f3 = PXep3_p[2][1]*fprime_e + PXep3_p[2][2]*2.*f_e*fprime_e;
            double f4 = PXep4_p[2][1]*fprime_e + PXep4_p[2][2]*2.*f_e*fprime_e;
            dB0de = dBdY[0][0] * f1 + 
                    dBdY[0][1] * f2 + 
                    dBdY[0][2] * f3 + 
                    dBdY[0][3] * f4;
            dB1de = dBdY[1][0] * f1 + 
                    dBdY[1][1] * f2 + 
                    dBdY[1][2] * f3 + 
                    dBdY[1][3] * f4;
            dB2de = dBdY[2][0] * f1 + 
                    dBdY[2][1] * f2 + 
                    dBdY[2][2] * f3 + 
                    dBdY[2][3] * f4;
            dB3de = dBdY[3][0] * f1 + 
                    dBdY[3][1] * f2 + 
                    dBdY[3][2] * f3 + 
                    dBdY[3][3] * f4;
        }
        return dB0de + dB1de*x + dB2de*x*x + dB3de*x*x*x;
    } else if (mode==3) { // Derivative wrt theta
        computecubicpars(0);
        // Now B[] contains the four parameters of the cubic
        double x = 0.5 + 4. * theta/thetamax;
        double dxdtheta = 4./thetamax;
        return (B[1] + B[2]*2.*x + B[3] * 3.*x*x) * dxdtheta;
    } 
    // If it gets here an invalid value of mode was passed
    // ---------------------------------------------------
    cout << "     Warning - invalid mode for cubic" << endl;
    warnings1++;
    return 0.; // This should not happen
}

// Function parametrizing muon content in gamma showers
// ----------------------------------------------------
double MFromG (double energy, double theta, double R, int mode, double dRdTh=0) {

    // Protect against out of range values
    // -----------------------------------
    if (R<Rmin) R = Rmin; 
    if (energy<0.1 || energy>10.) return 0.;
    if (theta<0. || theta>thetamax) return 0.;

    // Convert energy into the function we use in the interpolation
    // ------------------------------------------------------------
    double xe = 0.5+20.*(log(energy)-log_01)/logdif; // energy is in PeV
    double xe2 = xe*xe;

    // Interpolate the two parameters given the wanted energy and theta
    // ----------------------------------------------------------------
    double thisp0, thisp2;

    int ielow = (int)(energy/0.1)-1;
    int iehig = ielow + 1;
    double de = (energy-ielow*0.1-0.1)/0.1;
    int itlow = (int)(theta*99./thetamax);
    int ithig = itlow + 1;
    double dt = (theta - itlow*(thetamax/99.))/(thetamax/99.);
    // Handle boundaries first
    if (iehig==100) {
        if (ithig==100) { // upper edge of grid
            thisp0 = thisp0_mg[99][99];
            thisp2 = thisp2_mg[99][99];
        } else { // move on energy edge interpolating in theta
            thisp0 = thisp0_mg[99][itlow] + dt*(thisp0_mg[99][ithig]-thisp0_mg[99][itlow]);
            thisp2 = thisp2_mg[99][itlow] + dt*(thisp2_mg[99][ithig]-thisp2_mg[99][itlow]);
        }
    } else if (ithig==100) { // theta edge, interpolate in energy
        thisp0 = thisp0_mg[ielow][99] + de*(thisp0_mg[iehig][99]-thisp0_mg[ielow][99]);
        thisp2 = thisp2_mg[ielow][99] + de*(thisp2_mg[iehig][99]-thisp2_mg[ielow][99]);
    } else {
        // Do 2D interpolation
        double lowe0 = thisp0_mg[ielow][itlow] + dt*(thisp0_mg[ielow][ithig]-thisp0_mg[ielow][itlow]);
        double hige0 = thisp0_mg[iehig][itlow] + dt*(thisp0_mg[iehig][ithig]-thisp0_mg[iehig][itlow]);
        thisp0       = lowe0 + de*(hige0-lowe0);
        double lowe2 = thisp2_mg[ielow][itlow] + dt*(thisp2_mg[ielow][ithig]-thisp2_mg[ielow][itlow]);
        double hige2 = thisp2_mg[iehig][itlow] + dt*(thisp2_mg[iehig][ithig]-thisp2_mg[iehig][itlow]);
        thisp2       = lowe2 + de*(hige2-lowe2);
    }

    double flux = TankArea*0.02*thisp0*exp(-1.*pow(R,thisp2)); // function for muons. Note factor of 50
    if (mode==0) { // return function value
        if (flux>largenumber) return largenumber;
        if (flux<epsilon2) return 0.;
        if (flux!=flux) {
            cout << "Warning flux mg; E,T,R = " << energy << " " << theta << " " << R << endl;
            warnings2++;
            return 0.; // protect against nans
        }
        return flux;
    } else if (mode==1) { // return derivative with respect to R
        double dfluxdR = -flux * thisp2*pow(R,thisp2-1.);  
        if (dfluxdR!=dfluxdR) {
            cout << "Warning dfluxdR mg" << endl; 
            warnings2++;
            return 0.;
        }
        return dfluxdR; 
    } else if (mode==2) { // return derivative with respect to energy
        // Note, this is tricky: we are deriving 
        // f = p0(e)*exp(-R^p2(e)) over de.
        // This is like dh(g(f(e)))/de = h'(g(f))g'(f)f'(e),
        // So we get A*exp()*f(g(e))*f'(g(e))*g'(e)
        // In our case this is dp0/de * exp() + p0(e) * d(exp)/de =
        //                     dp0/de * f/p0 + f * [-R^(p2)] log(R) dp2/de 

        // Interpolate dthisp0de, dthisp2de - Handle boundaries first
        double dthisp0de, dthisp2de;
        if (iehig==100) {
            if (ithig==100) { // upper edge of grid
                dthisp0de = dthisp0de_mg[99][99];
                dthisp2de = dthisp2de_mg[99][99];
            } else { // move on energy edge interpolating in theta
                dthisp0de = dthisp0de_mg[99][itlow] + dt*(dthisp0de_mg[99][ithig]-dthisp0de_mg[99][itlow]);
                dthisp2de = dthisp2de_mg[99][itlow] + dt*(dthisp2de_mg[99][ithig]-dthisp2de_mg[99][itlow]);
            }
        } else if (ithig==100) { // theta edge, interpolate in energy
            dthisp0de = dthisp0de_mg[ielow][99] + de*(dthisp0de_mg[iehig][99]-dthisp0de_mg[ielow][99]);
            dthisp2de = dthisp2de_mg[ielow][99] + de*(dthisp2de_mg[iehig][99]-dthisp2de_mg[ielow][99]);
        } else {
            // Do 2D interpolation
            double lowe0 = dthisp0de_mg[ielow][itlow] + dt*(dthisp0de_mg[ielow][ithig]-dthisp0de_mg[ielow][itlow]);
            double hige0 = dthisp0de_mg[iehig][itlow] + dt*(dthisp0de_mg[iehig][ithig]-dthisp0de_mg[iehig][itlow]);
            dthisp0de    = lowe0 + de*(hige0-lowe0);
            double lowe2 = dthisp2de_mg[ielow][itlow] + dt*(dthisp2de_mg[ielow][ithig]-dthisp2de_mg[ielow][itlow]);
            double hige2 = dthisp2de_mg[iehig][itlow] + dt*(dthisp2de_mg[iehig][ithig]-dthisp2de_mg[iehig][itlow]);
            dthisp2de    = lowe2 + de*(hige2-lowe2);
        }

        double dfluxde = flux*(1./thisp0*dthisp0de -pow(R,thisp2)*log(R)*dthisp2de);
        if (dfluxde!=dfluxde) {
            cout << "Warning dfluxde mg; R = " << R << " thisp0, p2 = " << thisp0 << " " << thisp2 << " " << dthisp0de << " " << dthisp2de << endl; 
            warnings2++;
            return 0.;
        }
        return dfluxde;
    } else if (mode==3) { // return derivative with respect to theta
        // Interpolate dthisp0dth, dthisp2dth - Handle boundaries first
        double dthisp0dth, dthisp2dth;
        if (iehig==100) {
            if (ithig==100) { // upper edge of grid
                dthisp0dth = dthisp0dth_mg[99][99];
                dthisp2dth = dthisp2dth_mg[99][99];
            } else { // move on energy edge interpolating in theta
                dthisp0dth = dthisp0dth_mg[99][itlow] + dt*(dthisp0dth_mg[99][ithig]-dthisp0dth_mg[99][itlow]);
                dthisp2dth = dthisp2dth_mg[99][itlow] + dt*(dthisp2dth_mg[99][ithig]-dthisp2dth_mg[99][itlow]);
            }
        } else if (ithig==100) { // theta edge, interpolate in energy
            dthisp0dth = dthisp0dth_mg[ielow][99] + de*(dthisp0dth_mg[iehig][99]-dthisp0dth_mg[ielow][99]);
            dthisp2dth = dthisp2dth_mg[ielow][99] + de*(dthisp2dth_mg[iehig][99]-dthisp2dth_mg[ielow][99]);
        } else {
            // Do 2D interpolation
            double lowe0 = dthisp0dth_mg[ielow][itlow] + dt*(dthisp0dth_mg[ielow][ithig]-dthisp0dth_mg[ielow][itlow]);
            double hige0 = dthisp0dth_mg[iehig][itlow] + dt*(dthisp0dth_mg[iehig][ithig]-dthisp0dth_mg[iehig][itlow]);
            dthisp0dth   = lowe0 + de*(hige0-lowe0);
            double lowe2 = dthisp2dth_mg[ielow][itlow] + dt*(dthisp2dth_mg[ielow][ithig]-dthisp2dth_mg[ielow][itlow]);
            double hige2 = dthisp2dth_mg[iehig][itlow] + dt*(dthisp2dth_mg[iehig][ithig]-dthisp2dth_mg[iehig][itlow]);
            dthisp2dth   = lowe2 + de*(hige2-lowe2);
        }

        // Compute derivative. Careful, it's tricky too: We need to account for the dependence
        // of p0,p1,p2 on theta, but also of the dependence of R itself on theta!
        // -----------------------------------------------------------------------------------
        double dfluxdth   = flux*(1./thisp0*dthisp0dth 
                                  -thisp2*pow(R,thisp2-1)*dRdTh 
                                  -pow(R,thisp2)*log(R)*dthisp2dth);
        if (dfluxdth!=dfluxdth) {
            cout << "Warning dfluxdth mg" << endl; 
            warnings2++;
            return 0.;
        }
        return dfluxdth;
    } else {    
        return 0.;
    }
}

// Function parametrizing muon content in proton showers
// -----------------------------------------------------
double MFromP (double energy, double theta, double R, int mode, double dRdTh=0) {

    // Protect against out of range values
    // -----------------------------------
    if (R<Rmin) R = Rmin; 
    if (energy<0.1 || energy>10.) return 0.;
    if (theta<0. || theta>thetamax) return 0.;

    // Convert energy into the function we use in the interpolation
    // ------------------------------------------------------------
    double xe = 0.5+20.*(log(energy)-log_01)/logdif; // energy is in PeV
    double xe2 = xe*xe;

    // Interpolate the two parameters given the wanted theta
    // -----------------------------------------------------
    double thisp0, thisp2;

    int ielow = (int)(energy/0.1)-1;
    int iehig = ielow + 1;
    double de = (energy-ielow*0.1-0.1)/0.1;
    int itlow = (int)(theta/thetamax*99.);
    int ithig = itlow + 1;
    double dt = (theta - itlow*(thetamax/99.))/(thetamax/99.);
    // Handle boundaries first
    if (iehig==100) {
        if (ithig==100) { // upper edge of grid
            thisp0 = thisp0_mp[99][99];
            thisp2 = thisp2_mp[99][99];
        } else { // move on energy edge interpolating in theta
            thisp0 = thisp0_mp[99][itlow] + dt*(thisp0_mp[99][ithig]-thisp0_mp[99][itlow]);
            thisp2 = thisp2_mp[99][itlow] + dt*(thisp2_mp[99][ithig]-thisp2_mp[99][itlow]);
        }
    } else if (ithig==100) { // theta edge, interpolate in energy
        thisp0 = thisp0_mp[ielow][99] + de*(thisp0_mp[iehig][99]-thisp0_mp[ielow][99]);
        thisp2 = thisp2_mp[ielow][99] + de*(thisp2_mp[iehig][99]-thisp2_mp[ielow][99]);
    } else {
        // Do 2D interpolation
        double lowe0 = thisp0_mp[ielow][itlow] + dt*(thisp0_mp[ielow][ithig]-thisp0_mp[ielow][itlow]);
        double hige0 = thisp0_mp[iehig][itlow] + dt*(thisp0_mp[iehig][ithig]-thisp0_mp[iehig][itlow]);
        thisp0       = lowe0 + de*(hige0-lowe0);
        double lowe2 = thisp2_mp[ielow][itlow] + dt*(thisp2_mp[ielow][ithig]-thisp2_mp[ielow][itlow]);
        double hige2 = thisp2_mp[iehig][itlow] + dt*(thisp2_mp[iehig][ithig]-thisp2_mp[iehig][itlow]);
        thisp2       = lowe2 + de*(hige2-lowe2);
    }

    double flux = TankArea*0.02*thisp0*exp(-1.*pow(R,thisp2)); // function for muons. Note factor of 50 (different binning of original histos)
    if (mode==0) { // return function value
        if (flux>largenumber) return largenumber;
        if (flux<epsilon2) return 0.;
        if (flux!=flux) {
            cout << "Warning flux mp; E,T,R = " << energy << " " << theta << " " << R << endl;
            warnings2++;
            return 0.; // protect against nans
        }
        return flux;
    } else if (mode==1) { // return derivative with respect to R
        double dfluxdR = -flux * thisp2*pow(R,thisp2-1.);  
        if (dfluxdR!=dfluxdR) {
            cout << "Warning dfluxdR mp" << endl; 
            warnings2++;
            return 0.;
        }
        return dfluxdR; 
    } else if (mode==2) { // return derivative with respect to energy
        // Note, this is tricky: we are deriving A*exp(f(g(e))) over de.
        // This is like dh(g(f(e)))/de = h'(g(f))g'(f)f'(e),
        // So we get A*exp()*f(g(e))*f'(g(e))*g'(e)
        // That is A*exp()*R^p2*logR*dp2de = flux * R^p2 logR dp2de

        // Interpolate dthisp0de, dthisp2de - Handle boundaries first
        double dthisp0de, dthisp2de;
        if (iehig==100) {
            if (ithig==100) { // upper edge of grid
                dthisp0de = dthisp0de_mp[99][99];
                dthisp2de = dthisp2de_mp[99][99];
            } else { // move on energy edge interpolating in theta
                dthisp0de = dthisp0de_mp[99][itlow] + dt*(dthisp0de_mp[99][ithig]-dthisp0de_mp[99][itlow]);
                dthisp2de = dthisp2de_mp[99][itlow] + dt*(dthisp2de_mp[99][ithig]-dthisp2de_mp[99][itlow]);
            }
        } else if (ithig==100) { // theta edge, interpolate in energy
            dthisp0de = dthisp0de_mp[ielow][99] + de*(dthisp0de_mp[iehig][99]-dthisp0de_mp[ielow][99]);
            dthisp2de = dthisp2de_mp[ielow][99] + de*(dthisp2de_mp[iehig][99]-dthisp2de_mp[ielow][99]);
        } else {
            // Do 2D interpolation
            double lowe0 = dthisp0de_mp[ielow][itlow] + dt*(dthisp0de_mp[ielow][ithig]-dthisp0de_mp[ielow][itlow]);
            double hige0 = dthisp0de_mp[iehig][itlow] + dt*(dthisp0de_mp[iehig][ithig]-dthisp0de_mp[iehig][itlow]);
            dthisp0de    = lowe0 + de*(hige0-lowe0);
            double lowe2 = dthisp2de_mp[ielow][itlow] + dt*(dthisp2de_mp[ielow][ithig]-dthisp2de_mp[ielow][itlow]);
            double hige2 = dthisp2de_mp[iehig][itlow] + dt*(dthisp2de_mp[iehig][ithig]-dthisp2de_mp[iehig][itlow]);
            dthisp2de    = lowe2 + de*(hige2-lowe2);
        }

        double dfluxde = flux*(1./thisp0*dthisp0de -pow(R,thisp2)*log(R)*dthisp2de);
        if (dfluxde!=dfluxde) {
            cout << "Warning dfluxde mp" << endl; 
            warnings2++;
            return 0.;
        }
        return dfluxde;
    } else if (mode==3) { // return derivative with respect to theta
        // Interpolate dthisp0dth, dthisp2dth - Handle boundaries first
        double dthisp0dth, dthisp2dth;
        if (iehig==100) {
            if (ithig==100) { // upper edge of grid
                dthisp0dth = dthisp0dth_mp[99][99];
                dthisp2dth = dthisp2dth_mp[99][99];
            } else { // move on energy edge interpolating in theta
                dthisp0dth = dthisp0dth_mp[99][itlow] + dt*(dthisp0dth_mp[99][ithig]-dthisp0dth_mp[99][itlow]);
                dthisp2dth = dthisp2dth_mp[99][itlow] + dt*(dthisp2dth_mp[99][ithig]-dthisp2dth_mp[99][itlow]);
            }
        } else if (ithig==100) { // theta edge, interpolate in energy
            dthisp0dth = dthisp0dth_mp[ielow][99] + de*(dthisp0dth_mp[iehig][99]-dthisp0dth_mp[ielow][99]);
            dthisp2dth = dthisp2dth_mp[ielow][99] + de*(dthisp2dth_mp[iehig][99]-dthisp2dth_mp[ielow][99]);
        } else {
            // Do 2D interpolation
            double lowe0 = dthisp0dth_mp[ielow][itlow] + dt*(dthisp0dth_mp[ielow][ithig]-dthisp0dth_mp[ielow][itlow]);
            double hige0 = dthisp0dth_mp[iehig][itlow] + dt*(dthisp0dth_mp[iehig][ithig]-dthisp0dth_mp[iehig][itlow]);
            dthisp0dth   = lowe0 + de*(hige0-lowe0);
            double lowe2 = dthisp2dth_mp[ielow][itlow] + dt*(dthisp2dth_mp[ielow][ithig]-dthisp2dth_mp[ielow][itlow]);
            double hige2 = dthisp2dth_mp[iehig][itlow] + dt*(dthisp2dth_mp[iehig][ithig]-dthisp2dth_mp[iehig][itlow]);
            dthisp2dth   = lowe2 + de*(hige2-lowe2);
        }
        // Compute derivative. Careful, it's tricky too: We need to account for the dependence
        // of p0,p1,p2 on theta, but also of the dependence of R itself on theta!
        // -----------------------------------------------------------------------------------
        double dfluxdth   = flux*(1./thisp0*dthisp0dth 
                                  -thisp2*pow(R,thisp2-1)*dRdTh 
                                  -pow(R,thisp2)*log(R)*dthisp2dth);
        if (dfluxdth!=dfluxdth) {
            cout << "Warning dfluxdth mp" << endl; 
            warnings2++;
            return 0.;
        }
        return dfluxdth;
    } else {    
        return 0.;
    }
}

// Function parametrizing ele+gamma content in gamma showers
// ---------------------------------------------------------
double EFromG (double energy, double theta, double R, int mode, double dRdTh=0) {

    // Protect against out of range values
    // -----------------------------------
    if (R<Rmin) R = Rmin; 
    if (energy<0.1 || energy>10.) return 0.;
    if (theta<0. || theta>thetamax) return 0.;

    // Convert energy into the function we use in the interpolation
    // ------------------------------------------------------------
    double xe = 0.5+20.*(log(energy)-log_01)/logdif; // energy is in PeV
    double xe2 = xe*xe;

    // Interpolate the two parameters given the wanted theta
    // -----------------------------------------------------
    double thisp0, thisp1, thisp2;

    int ielow = (int)(energy/0.1)-1;
    int iehig = ielow + 1;
    double de = (energy-ielow*0.1-0.1)/0.1;
    int itlow = (int)(theta/thetamax*99.);
    int ithig = itlow + 1;
    double dt = (theta - itlow*(thetamax/99.))/(thetamax/99.);
    // Handle boundaries first
    if (iehig==100) {
        if (ithig==100) { // upper edge of grid
            thisp0 = thisp0_eg[99][99];
            thisp1 = thisp1_eg[99][99];
            thisp2 = thisp2_eg[99][99];
        } else { // move on energy edge interpolating in theta
            thisp0 = thisp0_eg[99][itlow] + dt*(thisp0_eg[99][ithig]-thisp0_eg[99][itlow]);
            thisp1 = thisp1_eg[99][itlow] + dt*(thisp1_eg[99][ithig]-thisp1_eg[99][itlow]);
            thisp2 = thisp2_eg[99][itlow] + dt*(thisp2_eg[99][ithig]-thisp2_eg[99][itlow]);
        }
    } else if (ithig==100) { // theta edge, interpolate in energy
        thisp0 = thisp0_eg[ielow][99] + de*(thisp0_eg[iehig][99]-thisp0_eg[ielow][99]);
        thisp1 = thisp1_eg[ielow][99] + de*(thisp1_eg[iehig][99]-thisp1_eg[ielow][99]);
        thisp2 = thisp2_eg[ielow][99] + de*(thisp2_eg[iehig][99]-thisp2_eg[ielow][99]);
    } else {
        // Do 2D interpolation
        double lowe0 = thisp0_eg[ielow][itlow] + dt*(thisp0_eg[ielow][ithig]-thisp0_eg[ielow][itlow]);
        double hige0 = thisp0_eg[iehig][itlow] + dt*(thisp0_eg[iehig][ithig]-thisp0_eg[iehig][itlow]);
        thisp0       = lowe0 + de*(hige0-lowe0);
        double lowe1 = thisp1_eg[ielow][itlow] + dt*(thisp1_eg[ielow][ithig]-thisp1_eg[ielow][itlow]);
        double hige1 = thisp1_eg[iehig][itlow] + dt*(thisp1_eg[iehig][ithig]-thisp1_eg[iehig][itlow]);
        thisp1       = lowe1 + de*(hige1-lowe1);
        double lowe2 = thisp2_eg[ielow][itlow] + dt*(thisp2_eg[ielow][ithig]-thisp2_eg[ielow][itlow]);
        double hige2 = thisp2_eg[iehig][itlow] + dt*(thisp2_eg[iehig][ithig]-thisp2_eg[iehig][itlow]);
        thisp2       = lowe2 + de*(hige2-lowe2);
    }

    double flux = TankArea*thisp0*exp(-thisp1*pow(R,thisp2)); 
    if (mode==0) { // return function value
        if (flux>largenumber) return largenumber;
        if (flux<epsilon2) return 0.;
        if (flux!=flux) {
            cout << "Warning flux eg; E,T,R = " << energy << " " << theta << " " << R << endl;
            warnings2++;
            return 0.; // protect against nans
        }
        return flux;
    } else if (mode==1) { // return derivative with respect to R
        double dfluxdR = -flux * thisp1*thisp2*pow(R,thisp2-1);  
        if (dfluxdR!=dfluxdR) {
            cout << "Warning dfluxdR eg" << endl; 
            warnings2++;
            return 0.;
        }
        return dfluxdR; 
    } else if (mode==2) { // return derivative with respect to energy
        // Note, this is tricky: we are deriving A*exp(f(g(e))) over de.
        // This is like dh(g(f(e)))/de = h'(g(f))g'(f)f'(e),
        // So we get A*exp()*f(g(e))*f'(g(e))*g'(e)
        // That is A*exp()*(-p1*R^p2)*logR*dp2de = flux * -p1*R^p2 logR dp2de
        // flux = A(e)*exp(-p1(e)*R^p2(e))
        // dflux/de = flux/A *dA/de + flux*[ -R^p2(e)*dp1/de -p1(e)R^p2(e) logR dp2de] 

        // Interpolate dthisp0de, dthisp2de - Handle boundaries first
        double dthisp0de, dthisp1de, dthisp2de;
        if (iehig==100) {
            if (ithig==100) { // upper edge of grid
                dthisp0de = dthisp0de_eg[99][99];
                dthisp1de = dthisp1de_eg[99][99];
                dthisp2de = dthisp2de_eg[99][99];
            } else { // move on energy edge interpolating in theta
                dthisp0de = dthisp0de_eg[99][itlow] + dt*(dthisp0de_eg[99][ithig]-dthisp0de_eg[99][itlow]);
                dthisp1de = dthisp1de_eg[99][itlow] + dt*(dthisp1de_eg[99][ithig]-dthisp1de_eg[99][itlow]);
                dthisp2de = dthisp2de_eg[99][itlow] + dt*(dthisp2de_eg[99][ithig]-dthisp2de_eg[99][itlow]);
            }
        } else if (ithig==100) { // theta edge, interpolate in energy
            dthisp0de = dthisp0de_eg[ielow][99] + de*(dthisp0de_eg[iehig][99]-dthisp0de_eg[ielow][99]);
            dthisp1de = dthisp1de_eg[ielow][99] + de*(dthisp1de_eg[iehig][99]-dthisp1de_eg[ielow][99]);
            dthisp2de = dthisp2de_eg[ielow][99] + de*(dthisp2de_eg[iehig][99]-dthisp2de_eg[ielow][99]);
        } else {
            // Do 2D interpolation
            double lowe0 = dthisp0de_eg[ielow][itlow] + dt*(dthisp0de_eg[ielow][ithig]-dthisp0de_eg[ielow][itlow]);
            double hige0 = dthisp0de_eg[iehig][itlow] + dt*(dthisp0de_eg[iehig][ithig]-dthisp0de_eg[iehig][itlow]);
            dthisp0de    = lowe0 + de*(hige0-lowe0);
            double lowe1 = dthisp1de_eg[ielow][itlow] + dt*(dthisp1de_eg[ielow][ithig]-dthisp1de_eg[ielow][itlow]);
            double hige1 = dthisp1de_eg[iehig][itlow] + dt*(dthisp1de_eg[iehig][ithig]-dthisp1de_eg[iehig][itlow]);
            dthisp1de    = lowe1 + de*(hige1-lowe1);
            double lowe2 = dthisp2de_eg[ielow][itlow] + dt*(dthisp2de_eg[ielow][ithig]-dthisp2de_eg[ielow][itlow]);
            double hige2 = dthisp2de_eg[iehig][itlow] + dt*(dthisp2de_eg[iehig][ithig]-dthisp2de_eg[iehig][itlow]);
            dthisp2de    = lowe2 + de*(hige2-lowe2);
        }

        double dfluxde = flux*(1./thisp0*dthisp0de -pow(R,thisp2)*dthisp1de -thisp1*pow(R,thisp2)*log(R)*dthisp2de);
        if (dfluxde!=dfluxde) {
            cout << "Warning dfluxde eg" << endl; 
            warnings2++;
            return 0.;
        }
        return dfluxde;
    } else if (mode==3) { // return derivative with respect to theta

        // Interpolate dthisp0dth, dthisp2dth - Handle boundaries first
        double dthisp0dth, dthisp1dth, dthisp2dth;
        if (iehig==100) {
            if (ithig==100) { // upper edge of grid
                dthisp0dth = dthisp0dth_eg[99][99];
                dthisp1dth = dthisp1dth_eg[99][99];
                dthisp2dth = dthisp2dth_eg[99][99];
            } else { // move on energy edge interpolating in theta
                dthisp0dth = dthisp0dth_eg[99][itlow] + dt*(dthisp0dth_eg[99][ithig]-dthisp0dth_eg[99][itlow]);
                dthisp1dth = dthisp1dth_eg[99][itlow] + dt*(dthisp1dth_eg[99][ithig]-dthisp1dth_eg[99][itlow]);
                dthisp2dth = dthisp2dth_eg[99][itlow] + dt*(dthisp2dth_eg[99][ithig]-dthisp2dth_eg[99][itlow]);
            }
        } else if (ithig==100) { // theta edge, interpolate in energy
            dthisp0dth = dthisp0dth_eg[ielow][99] + de*(dthisp0dth_eg[iehig][99]-dthisp0dth_eg[ielow][99]);
            dthisp1dth = dthisp1dth_eg[ielow][99] + de*(dthisp1dth_eg[iehig][99]-dthisp1dth_eg[ielow][99]);
            dthisp2dth = dthisp2dth_eg[ielow][99] + de*(dthisp2dth_eg[iehig][99]-dthisp2dth_eg[ielow][99]);
        } else {
            // Do 2D interpolation
            double lowe0 = dthisp0dth_eg[ielow][itlow] + dt*(dthisp0dth_eg[ielow][ithig]-dthisp0dth_eg[ielow][itlow]);
            double hige0 = dthisp0dth_eg[iehig][itlow] + dt*(dthisp0dth_eg[iehig][ithig]-dthisp0dth_eg[iehig][itlow]);
            dthisp0dth   = lowe0 + de*(hige0-lowe0);
            double lowe1 = dthisp1dth_eg[ielow][itlow] + dt*(dthisp1dth_eg[ielow][ithig]-dthisp1dth_eg[ielow][itlow]);
            double hige1 = dthisp1dth_eg[iehig][itlow] + dt*(dthisp1dth_eg[iehig][ithig]-dthisp1dth_eg[iehig][itlow]);
            dthisp1dth   = lowe1 + de*(hige1-lowe1);
            double lowe2 = dthisp2dth_eg[ielow][itlow] + dt*(dthisp2dth_eg[ielow][ithig]-dthisp2dth_eg[ielow][itlow]);
            double hige2 = dthisp2dth_eg[iehig][itlow] + dt*(dthisp2dth_eg[iehig][ithig]-dthisp2dth_eg[iehig][itlow]);
            dthisp2dth   = lowe2 + de*(hige2-lowe2);
        }
        // Compute derivative. Careful, it's tricky too: We need to account for the dependence
        // of p0,p1,p2 on theta, but also of the dependence of R itself on theta!
        // -----------------------------------------------------------------------------------
        double dfluxdth   = flux*(1./thisp0*dthisp0dth 
                                  -thisp1*thisp2*pow(R,thisp2-1.)*dRdTh 
                                  -pow(R,thisp2)*log(R)*dthisp2dth);
        if (dfluxdth!=dfluxdth) {
            cout << "Warning dfluxdth eg" << endl; 
            warnings2++;
            return 0.;
        }
        return dfluxdth;
    } else {    
        return 0.;
    }
}

// Function parametrizing ele+gamma content in proton showers
// ----------------------------------------------------------
double EFromP (double energy, double theta, double R, int mode, double dRdTh=0) {

    // Protect against out of range values
    // -----------------------------------
    if (R<Rmin) R = Rmin; 
    if (energy<0.1 || energy>10.) return 0.;
    if (theta<0. || theta>thetamax) return 0.;

    // Convert energy into the function we use in the interpolation
    // ------------------------------------------------------------
    double xe = 0.5+20.*(log(energy)-log_01)/logdif; // energy is in PeV
    double xe2 = xe*xe;

    // Interpolate the two parameters given the wanted theta
    // -----------------------------------------------------
    double thisp0, thisp1, thisp2;

    int ielow = (int)(energy/0.1)-1;
    int iehig = ielow + 1;
    double de = (energy-ielow*0.1-0.1)/0.1;
    int itlow = (int)(theta/thetamax*99.);
    int ithig = itlow + 1;
    double dt = (theta - itlow*(thetamax/99.))/(thetamax/99.);
    // Handle boundaries first
    if (iehig==100) {
        if (ithig==100) { // upper edge of grid
            thisp0 = thisp0_ep[99][99];
            thisp1 = thisp1_ep[99][99];
            thisp2 = thisp2_ep[99][99];
        } else { // move on energy edge interpolating in theta
            thisp0 = thisp0_ep[99][itlow]+ dt*(thisp0_ep[99][ithig]-thisp0_ep[99][itlow]);
            thisp1 = thisp1_ep[99][itlow]+ dt*(thisp1_ep[99][ithig]-thisp1_ep[99][itlow]);
            thisp2 = thisp2_ep[99][itlow]+ dt*(thisp2_ep[99][ithig]-thisp2_ep[99][itlow]);
        }
    } else if (ithig==100) { // theta edge, interpolate in energy
        thisp0 = thisp0_ep[ielow][99]+ de*(thisp0_ep[iehig][99]-thisp0_ep[ielow][99]);
        thisp1 = thisp1_ep[ielow][99]+ de*(thisp1_ep[iehig][99]-thisp1_ep[ielow][99]);
        thisp2 = thisp2_ep[ielow][99]+ de*(thisp2_ep[iehig][99]-thisp2_ep[ielow][99]);
    } else {
        // Do 2D interpolation
        double lowe0 = thisp0_ep[ielow][itlow]+ dt*(thisp0_ep[ielow][ithig]-thisp0_ep[ielow][itlow]);
        double hige0 = thisp0_ep[iehig][itlow]+ dt*(thisp0_ep[iehig][ithig]-thisp0_ep[iehig][itlow]);
        thisp0       = lowe0 + de*(hige0-lowe0);
        double lowe1 = thisp1_ep[ielow][itlow]+ dt*(thisp1_ep[ielow][ithig]-thisp1_ep[ielow][itlow]);
        double hige1 = thisp1_ep[iehig][itlow]+ dt*(thisp1_ep[iehig][ithig]-thisp1_ep[iehig][itlow]);
        thisp1       = lowe1 + de*(hige1-lowe1);
        double lowe2 = thisp2_ep[ielow][itlow]+ dt*(thisp2_ep[ielow][ithig]-thisp2_ep[ielow][itlow]);
        double hige2 = thisp2_ep[iehig][itlow]+ dt*(thisp2_ep[iehig][ithig]-thisp2_ep[iehig][itlow]);
        thisp2       = lowe2 + de*(hige2-lowe2);
    }

    double flux = TankArea*thisp0*exp(-thisp1*pow(R,thisp2)); 
    if (mode==0) { // return function value
        if (flux>largenumber) return largenumber;
        if (flux<epsilon2) return 0.;
        if (flux!=flux) {
            cout << "Warning flux ep; E,T,R = " << energy << " " << theta << " " << R << endl;
            warnings2++;
            return 0.; // protect against nans
        }
        return flux;
    } else if (mode==1) { // return derivative with respect to R
        double dfluxdR = -flux * thisp1*thisp2*pow(R,thisp2-1.);  
        if (dfluxdR!=dfluxdR) {
            cout << "Warning dfluxdR ep" << endl; 
            warnings2++;
            return 0.;
        }
        return dfluxdR; 
    } else if (mode==2) { // return derivative with respect to energy

        // Interpolate dthisp0de, dthisp2de - Handle boundaries first
        double dthisp0de, dthisp1de, dthisp2de;
        if (iehig==100) {
            if (ithig==100) { // upper edge of grid
                dthisp0de = dthisp0de_ep[99][99];
                dthisp1de = dthisp1de_ep[99][99];
                dthisp2de = dthisp2de_ep[99][99];
            } else { // move on energy edge interpolating in theta
                dthisp0de = dthisp0de_ep[99][itlow]+ dt*(dthisp0de_ep[99][ithig]-dthisp0de_ep[99][itlow]);
                dthisp1de = dthisp1de_ep[99][itlow]+ dt*(dthisp1de_ep[99][ithig]-dthisp1de_ep[99][itlow]);
                dthisp2de = dthisp2de_ep[99][itlow]+ dt*(dthisp2de_ep[99][ithig]-dthisp2de_ep[99][itlow]);
            }
        } else if (ithig==100) { // theta edge, interpolate in energy
            dthisp0de = dthisp0de_ep[ielow][99]+ de*(dthisp0de_ep[iehig][99]-dthisp0de_ep[ielow][99]);
            dthisp1de = dthisp1de_ep[ielow][99]+ de*(dthisp1de_ep[iehig][99]-dthisp1de_ep[ielow][99]);
            dthisp2de = dthisp2de_ep[ielow][99]+ de*(dthisp2de_ep[iehig][99]-dthisp2de_ep[ielow][99]);
        } else {
            // Do 2D interpolation
            double lowe0 = dthisp0de_ep[ielow][itlow]+ dt*(dthisp0de_ep[ielow][ithig]-dthisp0de_ep[ielow][itlow]);
            double hige0 = dthisp0de_ep[iehig][itlow]+ dt*(dthisp0de_ep[iehig][ithig]-dthisp0de_ep[iehig][itlow]);
            dthisp0de    = lowe0 + de*(hige0-lowe0);
            double lowe1 = dthisp1de_ep[ielow][itlow]+ dt*(dthisp1de_ep[ielow][ithig]-dthisp1de_ep[ielow][itlow]);
            double hige1 = dthisp1de_ep[iehig][itlow]+ dt*(dthisp1de_ep[iehig][ithig]-dthisp1de_ep[iehig][itlow]);
            dthisp1de    = lowe1 + de*(hige1-lowe1);
            double lowe2 = dthisp2de_ep[ielow][itlow]+ dt*(dthisp2de_ep[ielow][ithig]-dthisp2de_ep[ielow][itlow]);
            double hige2 = dthisp2de_ep[iehig][itlow]+ dt*(dthisp2de_ep[iehig][ithig]-dthisp2de_ep[iehig][itlow]);
            dthisp2de    = lowe2 + de*(hige2-lowe2);
        }

        // Note, this is tricky: we are deriving A*exp(f(g(e))) over de.
        // This is like dh(g(f(e)))/de = h'(g(f))g'(f)f'(e),
        // So we get A*exp()*f(g(e))*f'(g(e))*g'(e)
        // That is A*exp()*(-p1*R^p2)*logR*dp2de = flux * R^p2 logR dp2de
        // --------------------------------------------------------------
        double dfluxde = flux*(1./thisp0*dthisp0de -pow(R,thisp2)*dthisp1de -thisp1*pow(R,thisp2)*log(R)*dthisp2de);
        if (dfluxde!=dfluxde) {
            cout << "Warning dfluxde ep; R = " << R << " thisp0, p2 = " << thisp0 << " " << thisp2 << " " << dthisp0de << " " << dthisp2de << endl;  
            warnings2++;
            return 0.;
        }
        return dfluxde;
    } else if (mode==3) { // return derivative with respect to theta

        // Interpolate dthisp0dth, dthisp2dth - Handle boundaries first
        double dthisp0dth, dthisp1dth, dthisp2dth;
        if (iehig==100) {
            if (ithig==100) { // upper edge of grid
                dthisp0dth = dthisp0dth_ep[99][99];
                dthisp1dth = dthisp1dth_ep[99][99];
                dthisp2dth = dthisp2dth_ep[99][99];
            } else { // move on energy edge interpolating in theta
                dthisp0dth = dthisp0dth_ep[99][itlow] + dt*(dthisp0dth_ep[99][ithig]-dthisp0dth_ep[99][itlow]);
                dthisp1dth = dthisp1dth_ep[99][itlow] + dt*(dthisp1dth_ep[99][ithig]-dthisp1dth_ep[99][itlow]);
                dthisp2dth = dthisp2dth_ep[99][itlow] + dt*(dthisp2dth_ep[99][ithig]-dthisp2dth_ep[99][itlow]);
            }
        } else if (ithig==100) { // theta edge, interpolate in energy
            dthisp0dth = dthisp0dth_ep[ielow][99] + de*(dthisp0dth_ep[iehig][99]-dthisp0dth_ep[ielow][99]);
            dthisp1dth = dthisp1dth_ep[ielow][99] + de*(dthisp1dth_ep[iehig][99]-dthisp1dth_ep[ielow][99]);
            dthisp2dth = dthisp2dth_ep[ielow][99] + de*(dthisp2dth_ep[iehig][99]-dthisp2dth_ep[ielow][99]);
        } else {
            // Do 2D interpolation
            double lowe0 = dthisp0dth_ep[ielow][itlow] + dt*(dthisp0dth_ep[ielow][ithig]-dthisp0dth_ep[ielow][itlow]);
            double hige0 = dthisp0dth_ep[iehig][itlow] + dt*(dthisp0dth_ep[iehig][ithig]-dthisp0dth_ep[iehig][itlow]);
            dthisp0dth   = lowe0 + de*(hige0-lowe0);
            double lowe1 = dthisp1dth_ep[ielow][itlow] + dt*(dthisp1dth_ep[ielow][ithig]-dthisp1dth_ep[ielow][itlow]);
            double hige1 = dthisp1dth_ep[iehig][itlow] + dt*(dthisp1dth_ep[iehig][ithig]-dthisp1dth_ep[iehig][itlow]);
            dthisp1dth   = lowe1 + de*(hige1-lowe1);
            double lowe2 = dthisp2dth_ep[ielow][itlow] + dt*(dthisp2dth_ep[ielow][ithig]-dthisp2dth_ep[ielow][itlow]);
            double hige2 = dthisp2dth_ep[iehig][itlow] + dt*(dthisp2dth_ep[iehig][ithig]-dthisp2dth_ep[iehig][itlow]);
            dthisp2dth   = lowe2 + de*(hige2-lowe2);
        }
        // Compute derivative. Careful, it's tricky too: We need to account for the dependence
        // of p0,p1,p2 on theta, but also of the dependence of R itself on theta!
        // -----------------------------------------------------------------------------------
        double dfluxdth   = flux*(1./thisp0*dthisp0dth 
                                  -thisp1*thisp2*pow(R,thisp2-1.)*dRdTh 
                                  -pow(R,thisp2)*log(R)*dthisp2dth);
        if (dfluxdth!=dfluxdth) {
            cout << "Warning dfluxdth ep" << endl; 
            warnings2++;
            return 0.;
        }
        return dfluxdth;
    } else {    
        return 0.;
    }
}

// This function defines a layout which draws the word "MODE" on the ground with the detector positions
// ----------------------------------------------------------------------------------------------------
void DrawMODE(int idfirst, double x_lr, double y_lr, double xystep, double xstep, double ystep) {

    Nunits += 132;
    // M:
    for (int id=0; id<10; id++) {
        x[idfirst+id] = XDoffset + x_lr;
        y[idfirst+id] = YDoffset + y_lr+id*xystep;
    }
    for (int id=0; id<7; id++) {
        x[idfirst+id+10] = XDoffset + x_lr+id*xstep;
        y[idfirst+id+10] = YDoffset + y_lr+10*xystep-id*ystep;
    }
    for (int id=0; id<7; id++) {
        x[idfirst+id+17] = XDoffset + x_lr+7*xstep+id*xstep;
        y[idfirst+id+17] = YDoffset + y_lr+10*xystep-7*ystep+id*ystep;
    }
    for (int id=0; id<11; id++) {
        x[idfirst+id+24] = XDoffset + x_lr+14*xstep;
        y[idfirst+id+24] = YDoffset + y_lr+10*xystep-id*xystep;
    }
    // O:
    for (int id=0; id<10; id++) {
        x[idfirst+id+35] = XDoffset + x_lr+14*xstep+3*xystep;
        y[idfirst+id+35] = YDoffset + y_lr+id*xystep;
    }
    for (int id=0; id<7; id++) {
        x[idfirst+id+45] = XDoffset + x_lr+14*xstep+3*xystep+id*xystep;
        y[idfirst+id+45] = YDoffset + y_lr+10*xystep;
    }
    for (int id=0; id<10; id++) {
        x[idfirst+id+52] = XDoffset + x_lr+14*xstep+10*xystep;
        y[idfirst+id+52] = YDoffset + y_lr+10*xystep-id*xystep;
    }
    for (int id=0; id<7; id++) {
        x[idfirst+id+62] = XDoffset + x_lr+14*xstep+10*xystep-id*xystep;
        y[idfirst+id+62] = YDoffset + y_lr;
    }
    // D:
    for (int id=0; id<10; id++) {
        x[idfirst+id+69] = XDoffset + x_lr+14*xstep+13*xystep;
        y[idfirst+id+69] = YDoffset + y_lr+id*xystep;
    }
    for (int id=0; id<2; id++) {
        x[idfirst+id+79] = XDoffset + x_lr+14*xstep+13*xystep+id*xystep;
        y[idfirst+id+79] = YDoffset + y_lr+10*xystep;
    }
    for (int id=0; id<6; id++) {
        x[idfirst+id+81] = XDoffset + x_lr+14*xstep+15*xystep+id*xstep;
        y[idfirst+id+81] = YDoffset + y_lr+10*xystep-id*ystep;
    }
    for (int id=0; id<3; id++) {
        x[idfirst+id+87] = XDoffset + x_lr+19*xstep+15*xystep;
        y[idfirst+id+87] = YDoffset + y_lr+10*xystep-5*ystep-id*xystep;
    }
    for (int id=0; id<6; id++) {
        x[idfirst+id+90] = XDoffset + x_lr+19*xstep+15*xystep-id*xstep;
        y[idfirst+id+90] = YDoffset + y_lr+7*xystep-5*ystep-id*ystep;
    }
    for (int id=0; id<2; id++) {
        x[idfirst+id+96] = XDoffset + x_lr+13*xstep+15*xystep-id*xystep;
        y[idfirst+id+96] = YDoffset + y_lr;
    }
    // E:
    for (int id=0; id<10; id++) {
        x[idfirst+id+98] = XDoffset + x_lr+19*xstep+18*xystep;
        y[idfirst+id+98] = YDoffset + y_lr+id*xystep;
    }
    for (int id=0; id<8; id++) {
        x[idfirst+id+108] = XDoffset + x_lr+19*xstep+18*xystep+id*xystep;
        y[idfirst+id+108] = YDoffset + y_lr;
        x[idfirst+id+116] = XDoffset + x_lr+19*xstep+18*xystep+id*xystep;
        y[idfirst+id+116] = YDoffset + y_lr+5*xystep;
        x[idfirst+id+124] = XDoffset + x_lr+19*xstep+18*xystep+id*xystep;
        y[idfirst+id+124] = YDoffset + y_lr+10*xystep;
    }
}

// Define the current geometry by updating detector positions
// ----------------------------------------------------------
void DefineLayout (double detSpacing, double SpSt) {

    // We create a grid of detector positions.
    // We pave the xy space with a spiral from (0,0):
    // one step up, one right, two down, two left, three up, 
    // three right, four down, four left, etcetera.
    // The parameter SpSt widens the step progressively
    // from d to larger values. This allows to study layouts
    // with different density variations from center to periphery;
    // - d = spacing at start of spiral
    // - SpSt = rate of increase of spacing
    // - shape = shape of layout:
    //   0 = hexagonal, 
    //   1 = taxi
    //   2 = spiral 
    //   3 = circles 
    //   4 = random box 
    //   5 = word layout 
    //   6 = rectangle
    //   7 = four circles in a square
    //   8 = circular annulus, empty
    //   9 = double circular annulus
    //   101-115 = SWGO original array proposals (also redefines Nunits and TankArea)
    // ------------------------------------------------------------------------------
    x[0] = XDoffset;
    y[0] = YDoffset;
    int id = 1;
    if (shape==0) { // hexagonal grid
        double deltau = detSpacing;
        double deltav = detSpacing;
        double deltaz = detSpacing;
        int nstepsu = 1;
        int nstepsv = 1;
        int nstepsz = 1;
        double cos30 = sqrt(3.)/2.;
        double sin30 = 0.5;
        int parity = 1.;
        do {
            for (int is=0; is<nstepsu && id<Nunits; is++) {
                x[id] = x[id-1];
                y[id] = y[id-1] + deltau;
                id++;
            }
            deltau = -deltau;
            for (int is=0; is<nstepsv && id<Nunits; is++) {
                x[id] = x[id-1] + deltav*cos30;
                y[id] = y[id-1] + deltav*sin30;
                id++;
            }
            deltav = -deltav;
            for (int is=0; is<nstepsz && id<Nunits; is++) {
                x[id] = x[id-1] + deltaz*cos30;
                y[id] = y[id-1] - deltaz*sin30;
                id++;
            }
            deltaz = -deltaz;
            if (parity==-1) {
                nstepsv++;
            } else {
                nstepsu++;
                nstepsv++;
                nstepsz++;
            }
            parity *= -1;

            // After half cycle we increase the steps size
            // -------------------------------------------
            if (deltau>0) {
                deltau = deltau + SpSt;
            } else {
                deltau = deltau - SpSt;
            }
            if (deltav>0) {
                deltav = deltav + SpSt;
            } else {
                deltav = deltav - SpSt;
            }
            if (deltaz>0) {
                deltaz = deltaz + SpSt;
            } else {
                deltaz = deltaz - SpSt;
            }

        } while (id<Nunits); 
    } else if (shape==1) { // square grid
        int n_steps = 1;
        double deltax = detSpacing;
        double deltay = detSpacing;
        do {
            for (int is=0; is<n_steps && id<Nunits; is++) {
                x[id] = x[id-1];
                y[id] = y[id-1] + deltay;
                if (debug) cout << "id = " << id << " x,y = " << x[id] << "," <<  y[id] << endl;
                id++;
            }
            deltay = -deltay;
            for (int is=0; is<n_steps && id<Nunits; is++) {
                x[id] = x[id-1] + deltax;
                y[id] = y[id-1];
                if (debug) cout << "id = " << id << " x,y = " << x[id] << "," <<  y[id] << endl;
                id++;
            }
            deltax = -deltax;
            n_steps++;
            if (deltax>0) {
                deltax = deltax + SpSt;
            } else {
                deltax = deltax - SpSt;
            }
            if (deltay>0) {
                deltay = deltay + SpSt;
            } else {
                deltay = deltay - SpSt;
            }
        } while (id<Nunits); 
    } else if (shape==2) { // smooth spiral
        double delta = detSpacing;
        double angle0 = 1.; // better not be a submultiple of 2*pi if spiral_red is close to 1
        double angle = angle0;
        do {
            x[id] = x[id-1] + delta*cos(angle);
            y[id] = y[id-1] + delta*sin(angle);
            id++;
            angle0 = angle0*spiral_reduction;
            angle += angle0;
            delta = delta*SpSt; // step_increase;
            if (debug) cout << id << " " << angle << " " << delta << " " << cos(angle) << " " << sin(angle) << " " << x[id] << " " << y[id] << endl;
        } while (id<Nunits); 
    } else if (shape==3) {
        double r = detSpacing;
        do {
            double n = 6.*r/detSpacing;
            for (int ith=0; ith<n && id<Nunits; ith++) {
                double theta = ith*twopi/n;
                x[id] = XDoffset + r*cos(theta);
                y[id] = YDoffset + r*sin(theta);
                id++;
            }
            r = r + SpSt;
        } while (id<Nunits);
    } else if (shape==4) { // Random 2D box distribution
        for (int id=0; id<Nunits; id++) {
            double halfspan = detSpacing*sqrt(Nunits);
            x[id] = XDoffset + myRNG->Uniform(-halfspan,halfspan);
            y[id] = YDoffset + myRNG->Uniform(-halfspan,halfspan);
        }
    } else if (shape==5) { // Word layout
        int idfirst   = 0;
        double x_lr   = -410;
        double y_lr   = -50;
        double xystep = 10;
        double xstep  = 7;
        double ystep  = 7;
        Nunits = 0;
        DrawMODE(idfirst,x_lr,y_lr,xystep,xstep,ystep);
        idfirst   = 132;
        x_lr   = 40;
        y_lr   = -50;
        xystep = 10;
        xstep  = 7;
        ystep  = 7;
        DrawMODE(idfirst,x_lr,y_lr,xystep,xstep,ystep);
        //    Nunits = 132;
        idfirst = 264;
        x_lr    = -200;
        y_lr    = 250;
        xystep  = 10;
        xstep   = 7;
        ystep   = 7;
        DrawMODE(idfirst,x_lr,y_lr,xystep,xstep,ystep);
        idfirst = 396;
        // Nunits = 396;
        x_lr    = -200;
        y_lr    = -350;
        xystep  = 10;
        xstep   = 7;
        ystep   = 7;
        DrawMODE(idfirst,x_lr,y_lr,xystep,xstep,ystep);
        Nunits  = 528;
    } else if (shape==6) { // rectangle
        for (int id=0; id<Nunits; id++) {
            x[id] = XDoffset -detSpacing/2.+detSpacing*(id%2);
            y[id] = YDoffset -Nunits*0.25*detSpacing+detSpacing*(id/2);
        }
    } else if (shape==7) { // four circular densities in a square 
        double Qsize = sqrt(Nunits)*detSpacing; 
        double XVoffset, YVoffset;
        id = 0;
        for (int ivertex=0; ivertex<4; ivertex++) {
            XVoffset = -0.5*Qsize + (ivertex%2)*Qsize;
            YVoffset = -0.5*Qsize + (ivertex/2)*Qsize;
            double r = detSpacing;
            do {
                double n = 6.*r/detSpacing;
                for (int ith=0; ith<n && id<(ivertex+1)*Nunits/4; ith++) {
                    double theta = ith*twopi/n;
                    x[id] = XDoffset + r*cos(theta) + XVoffset;
                    y[id] = YDoffset + r*sin(theta) + YVoffset;
                    id++;
                }
                r = r + SpSt;
            } while (id<(ivertex+1)*Nunits/4);
        }

    } else if (shape==8) { // one annulus
        double r = detSpacing*Nunits/(2*pi);
        double n = Nunits;
        for (int id=0; id<n; id++) {
            double theta = id*twopi/n;
            x[id] = XDoffset + r*cos(theta);
            y[id] = YDoffset + r*sin(theta);
        }
    } else if (shape==9) { // two annuli
        if (Nunits%3!=0) {
            cout << "     For this config Nunits has to be multiple of 3. Changing it to next integer." << endl;
            Nunits = Nunits + 3-Nunits%3;
        }
        double r[2];
        r[0] = detSpacing*Nunits/(6.*pi);
        r[1] = detSpacing*Nunits/(3.*pi);
        double n[2];
        n[0] = Nunits/3;
        n[1] = Nunits-n[0];
        int id = 0;
        for (int ir=0; ir<2; ir++) {
            for (int i=0; i<n[ir]; i++) {
                double theta = i*twopi/n[ir];
                x[id] = XDoffset + r[ir]*cos(theta);
                y[id] = YDoffset + r[ir]*sin(theta);
                id++;
            }
        }
    } else if (shape==10) { // three annuli
        if (Nunits%3!=0) {
            cout << "     For this config Nunits has to be multiple of 3. Changing it to next integer." << endl;
            Nunits = Nunits + 3-Nunits%3;
        }
        double r[3];
        r[0] = detSpacing*Nunits/(6.*pi);
        r[1] = detSpacing*Nunits/(3.*pi);
        r[2] = detSpacing*Nunits/(2.*pi);
        double n[3];
        n[0] = Nunits/6;
        n[1] = Nunits/3;
        n[2] = Nunits/2;
        int id = 0;
        for (int ir=0; ir<3; ir++) {
            for (int i=0; i<n[ir]; i++) {
                double theta = i*twopi/n[ir];
                x[id] = XDoffset + r[ir]*cos(theta);
                y[id] = YDoffset + r[ir]*sin(theta);
                id++;
            }
        }
    } else if (shape>100 && shape<114) {
        // SWGO layouts. Coding is the following:
        // 101 == A1, 6589 entries
        // 102 == A2, 6631 entries
        // 103 == A3, 6823
        // 104 == A4, 6625
        // 105 == A5, 6541
        // 106 == A6, 6637
        // 107 == A7, 6571
        // 108 == B1, 4849
        // 109 == C1, 8371
        // 110 == D1, 3805
        // 111 == E1, 5461
        // 112 == E4, 5455
        // 113 == F1, 4681 

        // Set detector area to nominal value (we do not deal with aggregates of tanks here)
        // ---------------------------------------------------------------------------------
        TankArea = 3.61*pi;
        if (shape==101) Nunits = 6589;
        if (shape==102) Nunits = 6631;
        if (shape==103) Nunits = 6823;
        if (shape==104) Nunits = 6625;
        if (shape==105) Nunits = 6541;
        if (shape==106) Nunits = 6636;
        if (shape==107) Nunits = 6571;
        if (shape==108) Nunits = 4849;
        if (shape==109) Nunits = 8371;
        if (shape==110) Nunits = 3805;
        if (shape==111) Nunits = 5461;
        if (shape==112) Nunits = 5455;
        if (shape==113) Nunits = 4681;

#ifdef STANDALONE
        string detPath  = "/lustre/cmswork/dorigo/swgo/MT/Dets/";
#endif
#ifdef INROOT
        string detPath  = "./SWGO/Detectors/";
#endif
        ifstream detfile;
        char num[40];
        sprintf (num,"Layout_%d", shape);
        string detPositions = detPath  + num + ".txt";
        detfile.open(detPositions);
        double e;
        cout << endl;
        cout << "     ---------- Layout read in from config # " << shape << " ----------" << endl << endl;
        
        for (int id=0; id<Nunits; id++) {
            detfile >> e;
            if (e!=id) {
                cout << "Problem reading layout file" << endl;
                return;
            }
            detfile >> e;
            x[id] = e;
            detfile >> e;
            y[id] = e;
            // cout << "   Unit # " << id << ": x,y = " << x[id] << " " << y[id] << endl;
        }
        detfile.close();
        // cout << endl;        

        outfile << endl;
        outfile << "     ---------- Layout read in from config. # " << shape << " ----------" << endl << endl;
        /* 
        for (int id=0; id<Nunits; id++) {
            outfile << "    Unit # " << id << ": x,y = " << x[id] << " " << y[id] << endl;
        }
        outfile << "     -------------------------------------------------------" << endl << endl;
        outfile << endl;
        */

    } // end if shape

    // Define span x and y of generated showers to illuminate initial layout
    // ---------------------------------------------------------------------
    for (int id=0; id<Nunits; id++) {
        double thisr = sqrt(x[id]*x[id]+y[id]*y[id]);
        if (thisr>spanR) spanR = thisr;
    }
    return;
}

// Function that saves the layout data to file
// -------------------------------------------
void SaveLayout () {
#ifdef STANDALONE
    string detPath  = "/lustre/cmswork/dorigo/swgo/MT/Dets/";
#endif
#ifdef INROOT
    string detPath  = "./SWGO/Detectors/";
#endif
    ofstream detfile;
    std::stringstream sstr;
    char num[60];
    sprintf (num,"Nb=%d_Nu=%d_Ne=%d-%d_Sh=%d_Id=%d", Nbatch, Nunits, startEpoch, startEpoch+Nepochs, shape, indfile);
    sstr << "Layout_";
    string detPositions = detPath  + sstr.str() + num + ".txt";
    detfile.open(detPositions);
    for (int id=0; id<Nunits; id++) {
        detfile << x[id] << " " << y[id] << " " << endl;
    }
    detfile.close();
}
    
// Function that reads the layout data from file
// ---------------------------------------------
void ReadLayout () {
#ifdef STANDALONE
    string detPath  = "/lustre/cmswork/dorigo/swgo/MT/Dets/";
#endif
#ifdef INROOT
    string detPath  = "./SWGO/Detectors/";
#endif
    ifstream detfile;
    std::stringstream sstr;

    // Determine last file number to read
    // ----------------------------------
    indfile = -1;
    ifstream tmpfile;
    char num[100];
    do {
        if (indfile>-1) {
            tmpfile.close();
        }
        indfile++;
        sprintf (num, "Nb=%d_Nu=%d_Ne=%d-%d_Sh=%d_Id=%d", Nbatch, Nunits, startEpoch-Nepochs, startEpoch, shape, indfile);
        std::stringstream tmpstring;
        tmpstring << "Layout_";
        string tmpfilename = detPath + tmpstring.str() + num + ".txt";
        tmpfile.open(tmpfilename);
    } while (tmpfile.is_open());

    int thisindfile = indfile-1; // Must read the last one
    sprintf (num,"Nb=%d_Nu=%d_Ne=%d-%d_Sh=%d_Id=%d", Nbatch, Nunits, startEpoch-Nepochs, startEpoch, shape, thisindfile);
    sstr << "Layout_";
    string detPositions = detPath  + sstr.str() + num + ".txt";
    detfile.open(detPositions);
    double e;
    cout << endl;
    cout << "     ---------- Layout read in from previous run: ----------" << endl;
    cout << "                File is " << detPositions << endl;
    cout << endl;
    for (int id=0; id<Nunits; id++) {
        detfile >> e;
        x[id] = e;
        detfile >> e;
        y[id] = e;
        cout << "     Unit # " << id << ": x,y = " << x[id] << " " << y[id] << endl;
    }
    detfile.close();
    cout << endl;
    outfile << endl;
    outfile << "     ---------- Layout read in from previous run: ----------" << endl << endl;
    for (int id=0; id<Nunits; id++) {
        outfile << "    Unit # " << id << ": x,y = " << x[id] << " " << y[id] << endl;
    }
    outfile << "     -------------------------------------------------------" << endl << endl;
    outfile << endl;

    // Define span x and y of generated showers to illuminate initial layout
    // ---------------------------------------------------------------------
    for (int id=0; id<Nunits; id++) {
        double thisr = sqrt(x[id]*x[id]+y[id]*y[id]);
        if (thisr>spanR) spanR = thisr;
    }
    return;
}

// This function computes an approximation to the probability that a shower passes the
// condition N(units with >=1 observed particle) >= Ntrigger. See routine CheckProb
// and explanation in comments around computation of Pg, Pp. The function is called
// during GenerateShower to set the value of PActive[is].
// Note: if mode==0 the probability is computed (PActive[is])
// If instead mode==1 or 2, the derivative is returned of PActive[is]
// with respect to a movement of detector id. 
// -----------------------------------------------------------------------------------
double ProbTrigger (double SumProbs, int mode=0, int id=0, int is=0) {
    if (mode==0) {
        double sum = 0.;
        if (Ntrigger<maxNtrigger) {
            if (SumProbs>Ntrigger+3.*sqrt(Ntrigger+1)) return 1.; // good to 0.2%
            if (SumProbs<Ntrigger-3.*sqrt(Ntrigger+1)) return 0.; // good to 0.2%
            for (int k=0; k<Ntrigger; k++) {
                sum += Poisson (k,SumProbs); // exp(-SumProbs)*pow(SumProbs,k)/F[k]; // F is factorial
            }
        } else {
            cout << "Invalid Ntrigger" << endl;
            warnings6++;
            return 0.;
        }
        if (sum!=sum) {
            cout << "     Warning, trouble in ProbTrigger." << endl;
            warnings6++;
            return 1.;
        }
        if (sum>1.) return 0.;
        double p = 1.-sum;
        if (p<0.) return 0.;
        if (p>1.) return 1.;
        return p;
    } else if (mode==1) { // compute derivative wrt x[id]

        // Get dPA_dx, dPA_dy
        // To incorporate the contribution of PActive_m, in the calculation of dU/dx, we have made it part of the def of G.
        // This means we have a term to add to the calculations in ThreadFunction2, G/PActive_m * dPActive_m/dxi. 
        // The latter term can be computed as follows (PActive = 1 - sum(1:Ntr-1)):
        //     dPActive_m/dxi = -d/dxi [sum_{j=0}^{Ntr-1}(e^{-Sm}*Sm^j/j!)]
        // with
        //     Sm = Sum_{i=1}^Ndet [1-exp(-lambda_mu^i-lambda_e^i)_m]
        // We get
        //     dPActive_m/dxi = - Sum_{j=0}^{Ntr-1} 1/j! [e^{-Sm}*(Sm^j-j*Sm^(j-1))*dSm/dxi]
        // Now for dSm/dxi we have
        //     dSm/dxi = d/dxi [ Sum_{i=1}^{Ndet} (1-e^(-lambda_mu^i-lambda_e^i)_m))]
        //               = -d/dxi (e^[-lambda_mu^i-lambda_e^i]_m) =
        //               = e^[-lambda_mu^i-lambda_e^i]*(dlambda_mu^i/dxi + dlambda_e^i/dxi)
        // and the latter are computed in the flux routines.
        // Note that the lambdas are the true ones, as we compute PDFs with events that are
        // included in the sum if they pass the trigger, and they do if the true fluxes exceed
        // the threshold of Ntrigger detectors firing.
        // -----------------------------------------------------------------------------------
        // The calculation of the term due to dPActive/dxi is laborious, but we only need
        // to perform it if we are close to threshold, otherwise the derivative contr. is null
        // -----------------------------------------------------------------------------------
        double dPAm_dx = 0.;
        double xt   = TrueX0[is];
        double yt   = TrueY0[is];
        double tt   = TrueTheta[is];
        double pt   = TruePhi[is];
        double et   = TrueE[is];
        double ctt  = cos(tt);
        double Rim  =  EffectiveDistance (x[id],y[id],xt,yt,tt,pt,0);
        double dRdx = -EffectiveDistance (x[id],y[id],xt,yt,tt,pt,1); // wrt xi, so negative sign
        double lm, le, dlmdR, dledR;
        if (IsGamma[is]) {
            lm    = MFromG (et,tt,Rim,0)*ctt; 
            le    = EFromG (et,tt,Rim,0)*ctt;
            dlmdR = MFromG (et,tt,Rim,1)*ctt;
            dledR = EFromG (et,tt,Rim,1)*ctt;
        } else {
            lm    = MFromP (et,tt,Rim,0)*ctt; 
            le    = EFromP (et,tt,Rim,0)*ctt;
            dlmdR = MFromP (et,tt,Rim,1)*ctt;
            dledR = EFromP (et,tt,Rim,1)*ctt;
        }
        double dlmdx  = dlmdR*dRdx;
        double dledx  = dledR*dRdx;
        double dSm_dx = exp(-lm-le)*(dlmdx+dledx);
        for (int j=0; j<Ntrigger; j++) {
            if (j>0) {
                dPAm_dx += dSm_dx * (Poisson(j-1,SumProbs)-Poisson(j,SumProbs));            
            } else {
                dPAm_dx += - dSm_dx * Poisson(j,SumProbs);  // there is no term j*sm^j-1 if j=0          
            }
            // (exp(-SumProbs)*(pow(SumProbs,j)-j*pow(SumProbs,j-1))*dSm_dx)/F[j]; // F[j] is factorial
        }
        if (dPAm_dx!=dPAm_dx) {
            warnings4++;
            cout << "     Warning, trouble in dPActive/dx calculation." << endl;
            return 0.;
        }
        return dPAm_dx;
    } else if (mode==2) { // compute derivative wrt y[id]
        double dPAm_dy = 0.;
        double xt   = TrueX0[is];
        double yt   = TrueY0[is];
        double tt   = TrueTheta[is];
        double pt   = TruePhi[is];
        double et   = TrueE[is];
        double ctt  = cos(tt);
        double Rim  =  EffectiveDistance (x[id],y[id],xt,yt,tt,pt,0);
        double dRdy = -EffectiveDistance (x[id],y[id],xt,yt,tt,pt,2); // wrt yi
        double lm, le, dlmdR, dledR;
        if (IsGamma[is]) {
            lm    = MFromG (et,tt,Rim,0)*ctt; 
            le    = EFromG (et,tt,Rim,0)*ctt;
            dlmdR = MFromG (et,tt,Rim,1)*ctt;
            dledR = EFromG (et,tt,Rim,1)*ctt;
        } else {
            lm    = MFromP (et,tt,Rim,0)*ctt; 
            le    = EFromP (et,tt,Rim,0)*ctt;
            dlmdR = MFromP (et,tt,Rim,1)*ctt;
            dledR = EFromP (et,tt,Rim,1)*ctt;
        }
        double dlmdy  = dlmdR*dRdy;
        double dledy  = dledR*dRdy;
        double dSm_dy = exp(-lm-le)*(dlmdy+dledy);
        for (int j=0; j<Ntrigger; j++) {
            if (j>0) {
                dPAm_dy += dSm_dy * (Poisson(j-1,SumProbs)-Poisson(j,SumProbs));            
            } else {
                dPAm_dy += - dSm_dy * Poisson(j,SumProbs);  // there is no term j*sm^j-1 if j=0          
            }
        }
        if (dPAm_dy!=dPAm_dy) {
            warnings4++;
            cout << "     Warning, trouble in dPActive/dy calculation." << endl;
            return 0.;
        }
        return dPAm_dy;
    }
    cout << "     Warning, invalid mode in ProbTrigger, " << mode << endl;
    return 0.; // This should not happen
} 

// Get counts and times in detector units
// --------------------------------------
void GetCounts (int is) {

    int Ncount     = 0;
    SumProbgt1[is] = 0.;
    Active[is]     = false;
    PActive[is]    = 0.;
     for (int id=0; id<Nunits; id++) {   
        Nm[id][is] = 0.;
        Ne[id][is] = 0.;
        Tm[id][is] = 0.;
        Te[id][is] = 0.;
        double R = EffectiveDistance (x[id],y[id],TrueX0[is],TrueY0[is],TrueTheta[is],TruePhi[is],0);
        double ct = cos(TrueTheta[is]);
        double mug, eg, mup, ep;
        double Npexp = 0.;
        if (IsGamma[is]) {
            mug = MFromG (TrueE[is],TrueTheta[is],R,0) * ct;
            eg  = EFromG (TrueE[is],TrueTheta[is],R,0) * ct;
            if (mug>0.) Nm[id][is] = myRNG->Poisson(mug); // otherwise it remains zero
            if (eg>0.)  Ne[id][is] = myRNG->Poisson(eg);  // otherwise it remains zero
            // cout << " gamma E = " << TrueE[is] << " nm,ne = " << mug << " " << eg << endl;
            Npexp = mug+eg;
        } else {
            mup = MFromP (TrueE[is],TrueTheta[is],R,0) * ct;
            ep  = EFromP (TrueE[is],TrueTheta[is],R,0) * ct;
            if (mup>0.) Nm[id][is] = myRNG->Poisson(mup); // otherwise it remains zero
            if (ep>0.)  Ne[id][is] = myRNG->Poisson(ep);  // otherwise it remains zero
            // cout << " proton E = " << TrueE[is] << " nm,ne = " << mup << " " << ep << endl;
            Npexp = mup+ep;
        }
        SumProbgt1[is] += 1.-exp(-Npexp);     
        if (Nm[id][is]+Ne[id][is]>=1.) Ncount++;
        if (Nm[id][is]>0) Tm[id][is] = myRNG->Gaus (EffectiveTime(x[id],y[id],TrueX0[is],TrueY0[is],TrueTheta[is],TruePhi[is],0),sigma_time);
        if (Ne[id][is]>0) Te[id][is] = myRNG->Gaus (EffectiveTime(x[id],y[id],TrueX0[is],TrueY0[is],TrueTheta[is],TruePhi[is],0),sigma_time);
    }
    if (Ncount>=Ntrigger) Active[is] = true;
    PActive[is] = ProbTrigger (SumProbgt1[is],0);
    // cout << "PActive: " << is << " sp = " << SumProbgt1[is] << " PA= " << PActive[is] << endl;
    // With the above definitions, SumProbgt1 (and thus PActive) cannot be zero with Ncount>0, so 
    // Active[] is redundant (could check for PActive[]>0). However it is good to keep both, as
    // they represent two different things - observed and expected fluxes
    return;
}

// Set XY of showers in case they are not randomly resampled
// ---------------------------------------------------------
void SetShowersXY() {
    if (SetTo00) {
        for (int is=0; is<Nevents+Nbatch; is++) {
            TrueX0[is] = 0.;
            TrueY0[is] = 0.;
        }
    } else if (hexaShowers) {
        double r0 = sqrt(pow(spanR+Rslack,2)*pi/Nevents);
        TrueX0[0] = Xoffset;
        TrueY0[0] = Yoffset;
        int is   = 1;
        int n    = 6;
        double r = r0;
        do {
            for (int ith=0; ith<n && is<Nevents; ith++) {
                double phi = ith*twopi/n;
                TrueX0[is] = Xoffset + r*cos(phi);
                TrueY0[is] = Yoffset + r*sin(phi);
                if (debug) cout << is << " ith=" << ith << " r=" << r << " x,y= " << TrueX0[is] << " " << TrueY0[is] << endl;
                is++;
            }
            n += 6;
            r += r0;
        } while (is<Nevents);

        if (SameShowers) {
            for (int is=Nevents; is<Nevents+Nbatch; is++) {
                TrueX0[is] = TrueX0[is-Nevents];
                TrueY0[is] = TrueY0[is-Nevents];
            }   
        } else {
            // And now do the same for the Nbatch events
            // -----------------------------------------
            TrueX0[Nevents] = Xoffset;
            TrueY0[Nevents] = Yoffset;
            is = Nevents+1;
            n = 6;
            r = r0;
            do {
                for (int ith=0; ith<n && is<Nevents+Nbatch; ith++) {
                    double phi = ith*twopi/n;
                    TrueX0[is] = Xoffset + r*cos(phi);
                    TrueY0[is] = Yoffset + r*sin(phi);
                    if (debug) cout << is << " ith=" << ith << " r=" << r << " x,y= " << TrueX0[is] << " " << TrueY0[is] << endl;
                    is++;
                }
                n += 6;
                r += r0;
            } while (is<Nevents+Nbatch);
        }
    } else {
        // Below for a square grid of showers
        // ----------------------------------
        int side = sqrt(Nevents);
        for (int is=0; is<Nevents; is++) {
            TrueX0[is] = Xoffset -spanR -Rslack + 2.*(spanR+Rslack)*(is%side+0.5)/side;
            TrueY0[is] = Yoffset -spanR -Rslack + 2.*(spanR+Rslack)*(is/side+0.5)/side;
            if (debug) cout << is << " x,y= " << TrueX0[is] << " " << TrueY0[is] << endl;
        }

        // Same, for Nbatch events 
        // -----------------------
        side = sqrt(Nbatch);
        for (int is=Nevents; is<Nevents+Nbatch; is++) {
            TrueX0[is] = Xoffset -spanR -Rslack + 2.*(spanR+Rslack)*((is-Nevents)%side+0.5)/side;
            TrueY0[is] = Yoffset -spanR -Rslack + 2.*(spanR+Rslack)*((is-Nevents)/side+0.5)/side;
            if (debug) cout << is << " x,y= " << TrueX0[is] << " " << TrueY0[is] << endl;
        }
    }
}

// Generate shower and distribute particle signals in units
// --------------------------------------------------------
void GenerateShower (int is) {
 
    // Position of center of shower
    // Please remember that the generated distribution of showers must cover instrumented area
    // for all scanned configurations! We give some slack to the generated showers, such
    // that the system does not discover that the illuminated area has a step function
    // ---------------------------------------------------------------------------------------
    if (!fixShowerPos) { // otherwise x0, y0 are defined in setshowersxy routine
        if (SameShowers && is>=Nevents) {
            TrueX0[is] = TrueX0[is-Nevents];
            TrueY0[is] = TrueY0[is-Nevents];
        } else {
            if (hexaShowers) {
                // Hexagonal grid
                double DfromCenter = sqrt(myRNG->Uniform(0., pow(spanR+Rslack,2)));
                double phi = myRNG->Uniform(0.,twopi);
                TrueX0[is] = Xoffset + DfromCenter*cos(phi);  
                TrueY0[is] = Yoffset + DfromCenter*sin(phi);
            } else {
                // Square grid
                TrueX0[is] = Xoffset -(spanR+Rslack)+myRNG->Uniform(2.*(spanR+Rslack));
                TrueY0[is] = Yoffset -(spanR+Rslack)+myRNG->Uniform(2.*(spanR+Rslack));
            }
        }
    }    
    int Ncount     = 0;
    SumProbgt1[is] = 0.;
    Active[is]     = false;
    PActive[is]    = 0.;
    bool isGamma   = IsGamma[is];
    // Debugging tool: if requested, produce the same showers all the time
    if (SameShowers) {
        if (scanU) {
            TrueE[is] = 1.;
        } else {
            if (is<Nevents) {
                TrueE[is] = pow(pow(Emin,-Eslope)+(is+0.5)/Nevents*(pow(Emax,-Eslope)-pow(Emin,-Eslope)),-1./Eslope);
            } else {
                // TrueE[is] = TrueE[is-Nevents]; // recomputing it allows to run in multithreading even if SameShowers is on
                TrueE[is] = pow(pow(Emin,-Eslope)+(is-Nevents+0.5)/Nevents*(pow(Emax,-Eslope)-pow(Emin,-Eslope)),-1./Eslope);
             }
        }
        TrueTheta[is] = 0.;
        TruePhi[is]   = 0.;
        for (int id=0; id<Nunits; id++) {   
            Nm[id][is] = 0.;
            Ne[id][is] = 0.;
            Tm[id][is] = 0.;
            Te[id][is] = 0.;
            // TrueX0, TrueY0 are defined in main routine
            double trx0 = TrueX0[is];
            double try0 = TrueY0[is];
            // Handle definition in main routine to allow for multithreading 
            //  if (is>=Nevents) {
                // trx0 = TrueX0[is-Nevents];
                // try0 = TrueY0[is-Nevents];
            //  }
            double R = EffectiveDistance (x[id],y[id],trx0,try0,0.,0.,0);
            double mug, eg, mup, ep;
            double Npexp = 0.;
            if (isGamma) {
                mug = MFromG (TrueE[is],0.,R,0.); // no cos(truetheta) here as we are producing ortho showers if SameShowers is true
                eg  = EFromG (TrueE[is],0.,R,0.);
                if (mug>0.) Nm[id][is] = mug; // myRNG->Poisson(mug); 
                if (eg>0.)  Ne[id][is] = eg;  // myRNG->Poisson(eg); 
                Npexp = mug+eg;
            } else {
                mup = MFromP (TrueE[is],0.,R,0.);
                ep  = EFromP (TrueE[is],0.,R,0.);
                if (mup>0.) Nm[id][is] = mup; // myRNG->Poisson(mup); 
                if (ep>0.)  Ne[id][is] = ep;  // myRNG->Poisson(ep); 
                Npexp = mup+ep;
            }
            if (Npexp>=1.) Ncount++;
            if (Nm[id][is]>0) Tm[id][is] = myRNG->Gaus (0.,sigma_time);
            if (Ne[id][is]>0) Te[id][is] = myRNG->Gaus (0.,sigma_time);
            SumProbgt1[is] += 1.-exp(-Npexp);     
        } // end Nunit loop
        if (Ncount>=Ntrigger) Active[is] = true; // This shower is ok as far as observing 
        PActive[is] = ProbTrigger (SumProbgt1[is],0);
        // cout << "PActive: " << is << " " << P[is] << endl;
        return;
    }

    // We get the number density per m^2 of muons and other particles
    // as a function of R at the nominal detector position, for all detector units.
    // Note that the density is computed _at_ the detector center: for wide detectors this becomes an approximation.
    // Also note, this matrix does not depend on energy - it is regenerated
    // for every energy point (we only use it inside the ie loop in the code calling this function)
    // -------------------------------------------------------------------------------------------------------------

    // We want a PDF f(E) = A*E^-B. To normalize it in [Emin,Emax] we need A/B = 1./(Emin^-B-Emax^-B).
    // We get the integral of the PDF as F(E) = A/B (Emin^-B-E^-B) = (Emin^-B-E^-B)/(Emin^-B-Emax^-B).
    // Thus if we generate rnd as uniform, we get E = pow(Emin^-B + rnd * (Emax^-B-Emin^-B),(-1/B)).
    // -----------------------------------------------------------------------------------------------
    TrueE[is] = pow(pow(Emin,-Eslope)+myRNG->Uniform()*(pow(Emax,-Eslope)-pow(Emin,-Eslope)),-1./Eslope);
    
    // Define polar and azimuthal angle of shower
    // ------------------------------------------
    TruePhi[is]   = myRNG->Uniform(-pi,pi);
    if (OrthoShowers) {
        TrueTheta[is] = 0.;
    } else if (SlantedShowers) {
        TrueTheta[is] = pi/4.;
    } else {
        do {
            TrueTheta[is] = fabs(myRNG->Gaus()*pi/4.); // just a simple model 
        } while (TrueTheta[is]>=thetamax-epsilon);  // do not bother to simulate too tilted showers
    }

    // Now get actual readouts in the detectors 
    // ----------------------------------------
    GetCounts (is);
    return;
}

// Poisson likelihood for position and angle of shower, given Nmu, Ne counts in Nunit detectors
// --------------------------------------------------------------------------------------------
double ComputeLikelihood (int is, double x0, double y0, double theta, double phi, double energy, bool isgamma) {
    double logL = 0.;
    double ct   = cos(theta);
    for (int id=0; id<Nunits; id++) {
        float nm = Nm[id][is];
        float ne = Ne[id][is];
        double thisR = EffectiveDistance (x[id],y[id],x0,y0,theta,phi,0);
        double thisT = EffectiveTime     (x[id],y[id],x0,y0,theta,phi,0);
        double lambdaM, lambdaE;
        if (isgamma) { // gamma HYPOTHESIS
            lambdaM = MFromG (energy, theta, thisR, 0) * ct;
            lambdaE = EFromG (energy, theta, thisR, 0) * ct;
        } else {        // proton HYPOTHESIS        
            lambdaM = MFromP (energy, theta, thisR, 0) * ct;
            lambdaE = EFromP (energy, theta, thisR, 0) * ct;
        }
        if (lambdaM<=0.) { // zero total flux predictions - can happen if assumed X0,Y0 moved too far away
            if (nm>0.) return -largenumber;
        } else {
            logL -= lambdaM;
            logL += nm * log(lambdaM);
        }
        if (lambdaE<=0.) {
            if (ne>0.) return -largenumber;
        } else {
            logL -= lambdaE;
            logL += ne * log(lambdaE);
        }
        if (nm>0.) logL -= 0.5*pow((Tm[id][is]-thisT)/sigma_time,2.); // time is not defined otherwise
        if (ne>0.) logL -= 0.5*pow((Te[id][is]-thisT)/sigma_time,2.); // time is not defined otherwise
        // We omit the factorial term as we only use this likelihood in a ratio throughout, or only in comparison with fixed N.
    } // end id loop on Nunits
    return logL;
}

// Find most likely position of shower center by max likelihood
// ------------------------------------------------------------
double FitShowerParams (int is, bool GammaHyp) {

    // Initialize shower position at max flux
    // --------------------------------------
    double currentX0    = 0.;
    double currentY0    = 0.;
    double currentTheta = 0.5;
    double currentPhi   = 0.;
    double currentE     = 1.; // Just a first guess, midpoint of log_10 E between 100 TeV and 10 PeV

    // Preliminary assay of grid of points
    // -----------------------------------
    double maxlogL_in = -largenumber;
    double cX0, cY0, cT0, cP0, cE0;
    cX0 = currentX0;
    cY0 = currentY0;
    cT0 = currentTheta;
    cP0 = currentPhi;
    cE0 = currentE;
    int N_g = sqrt(1.*Ngrid);
    int N_p = 4;
    int N_t = 4;
    int N_e = NEgrid;
    if (usetrueXY) {
        N_g  = 1;
        currentX0 = TrueX0[is];
        currentY0 = TrueY0[is];
        cX0 = currentX0;
        cY0 = currentY0;
    }
    if (usetrueAngs) {
        N_p  = 1; 
        N_t  = 1;
        currentTheta = TrueTheta[is];
        currentPhi   = TruePhi[is];
        cT0 = currentTheta;
        cP0 = currentPhi;
    } 
    if (usetrueE) {
        N_e = 1;
        currentE = TrueE[is];
        cE0 = currentE;
    }
    // We might want to fit for parameters but force initialization to true values. We do it below
    // -------------------------------------------------------------------------------------------
    int currBitmap = initBitmap;
    if (currBitmap>=16) {
        currentX0 = TrueX0[is];
        currBitmap -= 16;
        cX0 = currentX0;
    }
    if (currBitmap>=8) {
        currentY0 = TrueY0[is];
        currBitmap -= 8;
        cY0 = currentY0;
    }
    if (currBitmap>=4) {
        currentTheta = TrueTheta[is];
        currBitmap -= 4;
        cT0 = currentTheta;
    } 
    if (currBitmap>=2) {
        currentPhi = TruePhi[is];
        currBitmap -= 2;
        cP0 = currentPhi;
    }
    if (currBitmap==1) {
        currentE = TrueE[is];
        cE0 = currentE; // need this in loop for xye below
    }
    //cout << is << " " << currBitmap << " " << cE0 << " " << TrueE[is] << endl;
    if (initBitmap%8==1 && !usetrueAngs) { // then we do not initialize theta, phi values to true ones.
        // We find suitable angles first, as they are identifiable even without precise E, X0, Y0
        // --------------------------------------------------------------------------------------
        for (int ip=0; ip<N_p; ip++) {
            if (N_p>1) cP0 = (-1.+(1.+2.*ip)/N_p)*pi;
            for (int it=0; it<N_t; it++) {
                if (N_t>1) cT0 = ((0.5+it)/N_t)*(halfpi-pi/8.);
                // Compute likelihood for this point
                double logL_in = ComputeLikelihood (is, currentX0, currentY0, cT0, cP0, currentE, GammaHyp);
                if (logL_in>maxlogL_in) {
                    currentTheta = cT0;
                    currentPhi   = cP0;
                    maxlogL_in   = logL_in;
                }
            }                
        }
    }
    
    // Now loop to find best point in X0,Y0,E
    // --------------------------------------
    if (initBitmap<8 && !usetrueXY) { // then we do not initialize x0,y0 to true values.
        maxlogL_in = -largenumber;
        for (int ie=0; ie<N_e; ie++) {
            if (N_e>1) cE0 = Einit[ie]; // if initBitmap%2=1 we have already initialized it 
            for (int ix=0; ix<N_g; ix++) {
                if (N_g>1) cX0 = (-1.+(1.+2.*ix)/N_g)*(spanR+Rslack);
                for (int iy=0; iy<N_g; iy++) {
                    if (N_g>1) cY0 = (-1.+(1.+2.*iy)/N_g)*(spanR+Rslack);
                    // Compute likelihood for this point
                    double logL_in = ComputeLikelihood (is, cX0, cY0, currentTheta, currentPhi, cE0, GammaHyp);
                    //if (GammaHyp==IsGamma[is] && is%100<2) cout << "ie,ix,iy = " << ie << " " << ix << " " << iy 
                    //                                         << " e,x,y =" << cE0 << " " << cX0 << " " << cY0 
                    //                                          << " et,xt,yt = " << TrueE[is] << 
                    //                                          " " << TrueX0[is] << " " << TrueY0[is] << " logL = " << logL_in << endl;
                    if (logL_in>maxlogL_in) {
                        currentX0    = cX0;
                        currentY0    = cY0;
                        currentE     = cE0;
                        maxlogL_in   = logL_in;
                    }
                }
            }
        }
    }
    //if (GammaHyp==IsGamma[is] && is%20==0) cout << "After init: " << currentX0 << " " << currentY0 << " true = " << TrueX0[is] << " " << TrueY0[is] << endl;

    // Declare and define variables used in loop below
    // -----------------------------------------------
    double dlogLdX0;
    double dlogLdY0;
    double dlogLdTh;
    double dlogLdPh;
    double dlogLdE;
    double logL           = 0.;
    double prevlogL       = 0.;
    double prev2logL      = 0.;
    double LearningRateX  = LRX;  
    double LearningRateY  = LRX; 
    double LearningRateTh = LRA;
    double LearningRatePh = LRA; 
    double LearningRateE  = LRE; 
    int istep  = 0;
    double m_x = 0.;
    double v_x = 0.;
    double m_y = 0.;
    double v_y = 0.;
    double m_t = 0.;
    double v_t = 0.;
    double m_p = 0.;
    double v_p = 0.;
    double m_e = 0.;
    double v_e = 0.;

    // Compute constant term in logL
    // -----------------------------
    double logLfix = 0.;
    for (int id=0; id<Nunits; id++) { 
        logLfix -= LogFactorial (Nm[id][is]);
        logLfix -= LogFactorial (Ne[id][is]);
    }

    // Loop to maximize logL and find X0, Y0, Theta, Phi, E of shower
    // --------------------------------------------------------------
    do {
        prev2logL  = prevlogL;
        prevlogL   = logL;
        logL       = logLfix; // We include this in the calculation, so that we can appraise the reconstruction performance
        dlogLdX0   = 0.;
        dlogLdY0   = 0.;
        dlogLdTh   = 0.;
        dlogLdPh   = 0.;
        dlogLdE    = 0.;
        double ct  = cos(currentTheta);

        // Sum contributions from all detectors to logL and derivatives
        // ------------------------------------------------------------
        for (int id=0; id<Nunits; id++) {

            float nm = Nm[id][is];
            float ne = Ne[id][is];
            float tm = Tm[id][is];
            float te = Te[id][is];

            // How far is this unit from assumed shower center, projected along direction?
            // ---------------------------------------------------------------------------
            double thisR = EffectiveDistance (x[id], y[id], currentX0, currentY0, currentTheta, currentPhi, 0);
            double thisT = EffectiveTime     (x[id], y[id], currentX0, currentY0, currentTheta, currentPhi, 0);
            double lambdaM0, lambdaE0;
            if (GammaHyp) { // gamma HYPOTHESIS
                lambdaM0 = MFromG (currentE, currentTheta, thisR, 0);
                lambdaE0 = EFromG (currentE, currentTheta, thisR, 0);
            } else {        // proton HYPOTHESIS        
                lambdaM0 = MFromP (currentE, currentTheta, thisR, 0);
                lambdaE0 = EFromP (currentE, currentTheta, thisR, 0);
            }

            double lambdaM = lambdaM0 * ct; // For tilted showers the flux is reduced, as the cross section of the tank is
            double lambdaE = lambdaE0 * ct; // (here we are not modeling the full cylinder, which would modify this simple picture)
            if (lambdaM<=0.) { // zero total flux predictions 
                if (nm>0.) logL -= largenumber;
            } else {
                logL += -lambdaM + nm*log(lambdaM);
            }
            if (lambdaE<=0.) {
                 if (ne>0.) logL -= largenumber;
            } else {
                logL += -lambdaE + ne*log(lambdaE);
            }
            // Add Gaussian term depending on detected arrival time t of particles in detector
            // -------------------------------------------------------------------------------
            if (nm>0.) logL -= 0.5 * pow((tm-thisT)/sigma_time,2.);
            if (ne>0.) logL -= 0.5 * pow((te-thisT)/sigma_time,2.);

            if (logL!=logL) {
                cout << " Warning: id = " << id << " E = " << currentE << " thisR = " << thisR << " logL = " << logL << " done with " 
                     << -lambdaM << " " << -lambdaE << " " << nm << " " << ne << " "
                     << -nm*log(lambdaM) << " " << -ne*log(lambdaE) << endl;
                cout << 0.5 * pow((tm-thisT)/sigma_time,2.) << " " << -lambdaE + ne*log(lambdaE) << " " << -lambdaM + nm * log(lambdaM) << endl;
                cout << currentX0 << " " << currentY0 << " " << currentTheta << " " << currentPhi << " " << currentE << endl;
                cout << x[id] << " " << y[id] << " " << tm << " " << te << " " << thisT << endl;
                warnings3++;
                SaveLayout();
                return 0.;
            }

            double dlM0dR, dlE0dR, dlogLdR;
            if (!usetrueXY || !usetrueAngs) { // these calcs are used in both cases
                if (GammaHyp) { // gamma HYP
                    dlM0dR = MFromG (currentE, currentTheta, thisR, 1);
                    dlE0dR = EFromG (currentE, currentTheta, thisR, 1);
                } else {        // proton HY1P
                    dlM0dR = MFromP (currentE, currentTheta, thisR, 1); 
                    dlE0dR = EFromP (currentE, currentTheta, thisR, 1); 
                }
                // Compute derivative of logL with respect to R and E.
                // Since log L = -lambda0*ct +N*log (lambda0*ct) + cost,
                // dlogL/dR = -ct*dl0dR + N/lambda0 * dl0dR .
                // The same calculation works for dlogL/dE
                // -------------------------------------------------------
                dlogLdR = - (dlM0dR + dlE0dR)*ct;
                if (lambdaM0>0.) {
                    dlogLdR += nm/lambdaM0 * dlM0dR;  
                }
                if (lambdaE0>0.)  {
                    dlogLdR += ne/lambdaE0 * dlE0dR;  
                }
            }
            if (!usetrueXY) {
                // Finally get dlogL/dx and dy from dlogLdR
                // ----------------------------------------
                double dRdX0 = EffectiveDistance (x[id], y[id], currentX0, currentY0, currentTheta, currentPhi, 1);
                double dRdY0 = EffectiveDistance (x[id], y[id], currentX0, currentY0, currentTheta, currentPhi, 2);
                dlogLdX0 += dRdX0 * dlogLdR;
                dlogLdY0 += dRdY0 * dlogLdR;
                double factorm = 0.;
                double factore = 0.;
                if (nm>0.) factorm = (tm-thisT)/(c0*sigma2_time);
                if (ne>0.) factore = (te-thisT)/(c0*sigma2_time);
                // Add contribution from dlogL/dthisT * dthisT/dX0,Y0. We get it from
                // dlogL/dthisT = 0.5*2*(t-thisT)/sigma2t  and  dthisT/dX = -(sin(theta)cos(phi)/c0,
                // ---------------------------------------------------------------------------------
                dlogLdX0 += (factorm+factore)*(-sin(currentTheta)*cos(currentPhi));
                dlogLdY0 += (factorm+factore)*(-sin(currentTheta)*sin(currentPhi));
            }
            if (!usetrueE) {
                double dlM0dE, dlE0dE;
                if (GammaHyp) { // gamma HYP
                    dlM0dE = MFromG (currentE, currentTheta, thisR, 2);
                    dlE0dE = EFromG (currentE, currentTheta, thisR, 2);
                } else {        // proton HYP
                    dlM0dE = MFromP (currentE, currentTheta, thisR, 2);
                    dlE0dE = EFromP (currentE, currentTheta, thisR, 2);
                }
                // As above, the cos(theta) factor only plays in the -1 part from the derivative dlogL/dlambda
                // -------------------------------------------------------------------------------------------
                dlogLdE -= (dlM0dE + dlE0dE)*ct;
                if (lambdaM0>0.) {
                    dlogLdE += nm/lambdaM0 * dlM0dE; 
                }
                if (lambdaE0>0.)  {
                    dlogLdE += ne/lambdaE0 * dlE0dE;
                }
            }
            if (!usetrueAngs) {
                // Also get dlogL/dtheta and dlogL/dphi
                // NB to handle the indirect dependence of lambdas on theta through R, we compute
                // this part first, and then use it in the dflux/dtheta calculations below (mode=3)
                // NB it would have been nicer to declare it static, but then multithreading would
                // mess it up...
                // --------------------------------------------------------------------------------
                double dRdTh = EffectiveDistance (x[id], y[id], currentX0, currentY0, currentTheta, currentPhi, 3);
                double dRdPh = EffectiveDistance (x[id], y[id], currentX0, currentY0, currentTheta, currentPhi, 4);

                double st    = sin(currentTheta);
                double sp    = sin(currentPhi);
                double cp    = cos(currentPhi);
                double dx    = x[id]-currentX0;
                double dy    = y[id]-currentY0;

                // Since 
                //    log L = -lambda0*ct +N*log (lambda0*ct) + cost,
                //    dlogL/dth = -ct*dl0dth + st*lambda0 + N/(lambda0*ct) * (dl0dth*ct - lambda0*st)
                // which becomes
                //    dlogL/dth = dl0dth*(N/lambda0-ct) + st*(lambda0-N/ct)
                // ----------------------------------------------------------------------------------
                double dlM0dth, dlE0dth;
                // Note that calculations below (mode=3) include use of dRdTh computed above.
                if (GammaHyp) { // gamma HYP
                    dlM0dth = MFromG (currentE, currentTheta, thisR, 3, dRdTh);
                    dlE0dth = EFromG (currentE, currentTheta, thisR, 3, dRdTh);
                } else {        // proton HYP
                    dlM0dth = MFromP (currentE, currentTheta, thisR, 3, dRdTh);
                    dlE0dth = EFromP (currentE, currentTheta, thisR, 3, dRdTh);
                }
                if (lambdaM0>0.) dlogLdTh += (nm/lambdaM0-ct) * dlM0dth + st*(lambdaM0-nm/ct);
                if (lambdaE0>0.) dlogLdTh += (ne/lambdaE0-ct) * dlE0dth + st*(lambdaE0-ne/ct); 

                dlogLdPh += dRdPh * dlogLdR;

                // Contributions from dL/dthisT * dthisT/dTh, dthisT/dPh:
                // ------------------------------------------------------
                if (nm>0.) {
                    dlogLdTh += (tm-thisT)/sigma2_time *ct*(cp*dx+sp*dy)/c0;
                    dlogLdPh += (tm-thisT)/sigma2_time *st*(-sp*dx+cp*dy)/c0;
                }
                if (ne>0.) {
                    dlogLdTh += (te-thisT)/sigma2_time *ct*(cp*dx+sp*dy)/c0;
                    dlogLdPh += (te-thisT)/sigma2_time *st*(-sp*dx+cp*dy)/c0;
                }
            }
        } // end id loop on Nunits

        // Take a step in X0, Y0
        // ---------------------
        if (!usetrueXY) { 
            // ADAM GD rule:
            // -------------
            m_x = beta1 * m_x + (1.-beta1)*dlogLdX0;
            v_x = beta2 * v_x + (1.-beta2)*pow(dlogLdX0,2);
            double m_x_hat = m_x/(1.-powbeta1[istep+1]);
            double v_x_hat = v_x/(1.-powbeta2[istep+1]);
            double incr = LearningRateX * m_x_hat / (sqrt(v_x_hat)+epsilon); 
            if (fabs(incr)>epsilon) currentX0 += incr;
            m_y = beta1 * m_y + (1.-beta1)*dlogLdY0;
            v_y = beta2 * v_y + (1.-beta2)*pow(dlogLdY0,2);
            double m_y_hat = m_y/(1.-powbeta1[istep+1]);
            double v_y_hat = v_y/(1.-powbeta2[istep+1]);
            incr = LearningRateY * m_y_hat / (sqrt(v_y_hat)+epsilon);
            if (fabs(incr)>epsilon) currentY0 += incr;
        }

        // Also take a step in theta and phi
        // ---------------------------------
        if (!usetrueAngs) {
            // ADAM GD rule:
            // -------------
            m_t = beta1 * m_t + (1.-beta1)*dlogLdTh;
            v_t = beta2 * v_t + (1.-beta2)*pow(dlogLdTh,2);
            double m_t_hat = m_t/(1.-powbeta1[istep+1]);
            double v_t_hat = v_t/(1.-powbeta2[istep+1]);
            double incr = LearningRateTh * m_t_hat / (sqrt(v_t_hat)+epsilon);
            if (fabs(incr)>epsilon) currentTheta += incr;
            m_p = beta1 * m_p + (1.-beta1)*dlogLdPh;
            v_p = beta2 * v_p + (1.-beta2)*pow(dlogLdPh,2);
            double m_p_hat = m_p/(1.-powbeta1[istep+1]);
            double v_p_hat = v_p/(1.-powbeta2[istep+1]);
            incr = LearningRatePh * m_p_hat / (sqrt(v_p_hat)+epsilon);
            if (fabs(incr)>epsilon)     currentPhi += incr;
            if (currentTheta>=thetamax) currentTheta = thetamax-epsilon; // hard reset if hitting boundary
            if (currentTheta<=0.)       currentTheta = epsilon;              // hard reset if hitting boundary
        }

        // And a step in E
        // ---------------
        if (!usetrueE) {
            // ADAM GD rule:
            // -------------
            m_e = beta1 * m_e + (1.-beta1)*dlogLdE;
            v_e = beta2 * v_e + (1.-beta2)*pow(dlogLdE,2);
            double m_e_hat = m_e/(1.-powbeta1[istep+1]);
            double v_e_hat = v_e/(1.-powbeta2[istep+1]);
            double incr = LearningRateE * m_e_hat / (sqrt(v_e_hat)+epsilon); 
            if (fabs(incr)>epsilon) currentE += incr;
            if (currentE<0.1) currentE = 0.1; // very loose requirement on reconstruction
            if (currentE>10.) currentE = 10.;
        }
        istep++;

        /*if ( istep%10==0 && (IsGamma[is] && GammaHyp || (!IsGamma[is] && !GammaHyp))) {
            cout << "  istep = " << istep << " logL = " << logL << " xt,yt = " << TrueX0[is] << "," << TrueY0[is] << " xm,ym = " << currentX0 << "," << currentY0 
                 << " Et = " << TrueE[is] << " Em = " << currentE << " Tt,Pt = " << TrueTheta[is] << "," << TruePhi[is] << " Tm,Pm = " << currentTheta << "," << currentPhi << endl;
        }*/

    } while (istep<maxNsteps && (istep<3 || fabs(logL-prevlogL)>logLApprox || fabs(prevlogL-prev2logL)>logLApprox)); // set it to 0.5 by default 

    NumAvgSteps+=istep;
    DenAvgSteps++;

#ifdef FEWPLOTS
    if (IsGamma[is] && GammaHyp || (!IsGamma[is] && !GammaHyp)) {
        NumStepsvsxy->Fill(sqrt(pow(TrueX0[is],2)+pow(TrueY0[is],2)),TrueE[is],(double)istep);
        NumStepsvsxyN->Fill(sqrt(pow(TrueX0[is],2)+pow(TrueY0[is],2)),TrueE[is]);
    }
#endif
#ifdef PLOTS
    if (IsGamma[is] && GammaHyp) NumStepsg->Fill(TrueE[is],(double)istep);
    if (!IsGamma[is] && !GammaHyp) NumStepsp->Fill(TrueE[is],(double)istep); 
#endif
    /*if ( (IsGamma[is] && GammaHyp || (!IsGamma[is] && !GammaHyp))) {
        cout << "  istep = " << istep << " logL = " << logL << " xt,yt = " << TrueX0[is] << "," << TrueY0[is] << " xm,ym = " << currentX0 << "," << currentY0 
                << " Et = " << TrueE[is] << " Em = " << currentE << " Tt,Pt = " << TrueTheta[is] << "," << TruePhi[is] << " Tm,Pm = " << currentTheta << "," << currentPhi << endl;
    }*/

    if (currentPhi>pi) currentPhi  -= twopi;
    if (currentPhi<-pi) currentPhi += twopi;
    if (debug) {
        if (IsGamma[is] && GammaHyp && Active[is]) {
            cout << " x,y = " << currentX0 << "," << currentY0 << " E = " << currentE << " true E = " << TrueE[is];
            if (SameShowers && is>=Nevents) cout << " Previous reco E = " << Emeas[is-Nevents][0];
            cout << endl;
        }
    }
    //if (!usetrueAngs) cout << "true th = " << TrueTheta[is] << " meas = " << currentTheta << " true ph = " << TruePhi[is] << " meas = " << currentPhi << endl;
    //if (!usetrueXY)   cout << "true x = " << TrueX0[is] << " meas = " << currentX0 << " true y = << " << TrueY0[is] << " meas = " << currentY0 << endl;

    // Now we have the estimates of X0, Y0, and the logLR at max for event is 
    // ----------------------------------------------------------------------
#ifdef PLOTS
    if (GammaHyp && IsGamma[is] && Active[is]) { // gamma
        double delta = currentX0-TrueX0[is];
        if (fabs(delta)<maxdxy) {
            DXG->Fill(delta);
        } else {
            if (delta>0.) DXG->Fill(maxdxy-epsilon);
            if (delta<0.) DXG->Fill(-maxdxy+epsilon);
        }
        delta = currentY0-TrueY0[is];
        if (fabs(delta)<maxdxy) {
            DYG->Fill(delta);
        } else {
            if (delta>0.) DYG->Fill(maxdxy-epsilon);
            if (delta<0.) DYG->Fill(-maxdxy+epsilon);
        }
        DTHG->Fill(currentTheta-TrueTheta[is]);
        DTHGvsT->Fill(currentTheta-TrueTheta[is],TrueTheta[is]);
        double dp = currentPhi - TruePhi[is];
        if (dp>2.*pi)  dp -= twopi;
        if (dp<-2.*pi) dp += twopi;
        DPHG->Fill(dp);
        DEG->Fill(fabs(currentE-TrueE[is])/TrueE[is]);
    } else if (!GammaHyp && !IsGamma[is] && Active[is]) { // proton
        double delta = currentX0-TrueX0[is];
        if (fabs(delta)<maxdxy) {
            DXP->Fill(delta);
        } else { 
            if (delta>0.) DXP->Fill(maxdxy-epsilon);
            if (delta<0.) DXP->Fill(-maxdxy+epsilon);
        }
        delta = currentY0-TrueY0[is];
        if (fabs(delta)<maxdxy) {
            DYP->Fill(delta);
        } else {
            if (delta>0.) DYP->Fill(maxdxy-epsilon);
            if (delta<0.) DYP->Fill(-maxdxy+epsilon);
        }
        DTHP->Fill(currentTheta-TrueTheta[is]);
        DTHPvsT->Fill(currentTheta-TrueTheta[is],TrueTheta[is]);
        double dp = currentPhi - TruePhi[is];
        if (dp>2.*pi)  dp -= twopi;
        if (dp<-2.*pi) dp += twopi;
        DPHP->Fill(dp);
        DEP->Fill(fabs(currentE-TrueE[is])/TrueE[is]);
    } 
#endif
    if (GammaHyp) {
        x0meas[is][0] = currentX0;
        y0meas[is][0] = currentY0;
        thmeas[is][0] = currentTheta;
        phmeas[is][0] = currentPhi;
        Emeas[is][0]  = currentE;
    } else {
        x0meas[is][1] = currentX0;
        y0meas[is][1] = currentY0;
        thmeas[is][1] = currentTheta;
        phmeas[is][1] = currentPhi;
        Emeas[is][1]  = currentE;
    }
    if (logL!=logL) {
        cout << "Problems with logL, return -largenumber" << endl;
        logL = -largenumber;
        SaveLayout();
        return 0.;
    }
    //if (IsGamma[is] && GammaHyp || (!IsGamma[is] && !GammaHyp)) cout << is << " " << logL << " " << prevlogL << " " << istep << endl;
    return logL;
}


void FindLogLR_new (int is) {

    // Try delta method
    // ----------------
    // 1. find vector of gradients of the LLR versus all model parameters (5 for photons, 5 for protons)
    // 2. find inverse of hessian 10x10 matrix
    // 3. compute an estimate of the variance of the LLR by  variance = grad(LLR)^T * Hessian^-1 * grad(LLR) 
    //
    // The observable quantities for each detector id=1...Ndet are static floats:
    //   Nm[id][is], Ne[id][is], Tm[id][is], Te[id][is]
    // We have estimated parameters under the two hypotheses also as static doubles:
    //   double emg   = Emeas[is][0];
    //   double emp   = Emeas[is][1];
    //   double x0g   = x0meas[is][0];
    //   double x0p   = x0meas[is][1];
    //   double y0g   = y0meas[is][0];
    //   double y0p   = y0meas[is][1];
    //   double thmg  = thmeas[is][0];
    //   double thmp  = thmeas[is][1];
    //   double phmg  = phmeas[is][0];
    //   double phmp  = phmeas[is][1];
    // 
    // You can check in the routine FindShowerPos() how the likelihood is computed. There,  
    // the maximization is done by computing the first derivatives of the likelihood over the five
    // parameters. Try solving the problem by yourself first though, so that you double check my
    // calculations!
    // -----------------------------------------------------------------------------------------------------







    /*double conts[25];
    for (int i=0; i<5; i++)
    Double_t tmp[numberOfLines*numberOfColumns]; 
    for(int i=0;i<numberOfLines;i++) { 
        for(int j=0;j<numberOfColumns;j++) 
            tmp[i*numberOfColumns+j]=getValue(i,j); 
        }
    } 
    TMatrixD mat(numberOfLines,numberOfColumns,tmp); 
    mat.Invert(); //mat.InvertFast(); 
    if(!mat.IsValid()) return matrixNxM(0,0); 
    matrixNxM ret(numberOfColumns,numberOfLines); 
    for(int i=0;i<numberOfLines;i++) { 
        for(int j=0;j<numberOfColumns;j++) ret.setValue(i,j,mat[i][j]); 
    } return ret;

    Double_t det2;
    TMatrixD H2 = H_square;
    H2.Invert(&det2);

    TMatrixD U2(H2,TMatrixD::kMult,H_square);
    TMatrixDDiag diag2(U2); diag2 = 0.0;
    const Double_t U2_max_offdiag = (U2.Abs()).Max();
    std::cout << "  Maximum off-diagonal = " << U2_max_offdiag << std::endl;
    std::cout << "  Determinant          = " << det2 << std::endl;
    */
}

// Compute log likelihood ratio test statistic for one shower, by
// finding max value vs X0,Y0,theta,phi of shower for both hypotheses,
// and its variance. The variance is computed by sampling and with an
// analytic approximation, depending on SampleT. The routine also computes
// the derivatives of sigma2 over dx and dy
// -----------------------------------------------------------------------
void FindLogLR (int is) { 

    // Compute the LRT
    // ---------------
    double logLG = FitShowerParams (is,true);  // Find shower position by max lik of gamma hypothesis
    double logLP = FitShowerParams (is,false); // Find shower position by max lik of proton hypothesis
    logLRT[is] = logLG-logLP; 

#ifdef FEWPLOTS
        if (IsGamma[is] && Active[is]) {
            DE->Fill(log(TrueE[is])/log_10,fabs(Emeas[is][0]-TrueE[is])/TrueE[is]);
            // Decide if logL is too low
            //if (TrueE[is]>em0) {
            //    logLvsdr->Fill(logLG,log(TrueE[is]-em0)/TrueE[is]);
            //} else {
            //    logLvsdr->Fill(logLG,-log(em0-TrueE[is])/TrueE[is]);
            //}
        }
#endif

    double sigmaT2_approx = 0.;
    double sigmaT2_sample = 0.; 
    double sigmaT2_deltam = 0.;

    if (SampleT) {
        float Nmstore[maxUnits];
        float Nestore[maxUnits];
        float Tmstore[maxUnits];
        float Testore[maxUnits];
        double X0st[2], Y0st[2], Thst[2], Phst[2], Est[2];
        X0st[0] = x0meas[is][0]; 
        Y0st[0] = y0meas[is][0]; 
        Thst[0] = thmeas[is][0];
        Phst[0] = phmeas[is][0];
        Est[0]  = Emeas[is][0];
        X0st[1] = x0meas[is][1]; 
        Y0st[1] = y0meas[is][1]; 
        Thst[1] = thmeas[is][1];
        Phst[1] = phmeas[is][1];
        Est[1]  = Emeas[is][1];

        // Store observed values
        // ---------------------
        for (int id=0; id<Nunits; id++) {
            Nmstore[id] = Nm[id][is];
            Nestore[id] = Ne[id][is];
            Tmstore[id] = Tm[id][is];
            Testore[id] = Te[id][is];
        }

        // Use repeated sampling to estimate the variance of T
        // ---------------------------------------------------
        double sumT  = 0.;
        double sumT2 = 0.;
        for (int irep=0; irep<Nrep; irep++) {
    
            // Regenerate counts and times
            // ---------------------------
            GetCounts (is);
            double logLGi = FitShowerParams (is,true);  // Find shower position by max lik of gamma hypothesis, 1= fluctuate nm,ne,tm,te
            double logLPi = FitShowerParams (is,false); // Find shower position by max lik of proton hypothesis,1= fluctuate nm,ne,tm,te
            double T = logLGi-logLPi; 
            sumT += T; 
            sumT2+= T*T;
        }
        sigmaT2_sample = sumT2/Nrep - pow(sumT/Nrep,2);

        // Restore original observed values
        // --------------------------------
        for (int id=0; id<Nunits; id++) {
            Nm[id][is] = Nmstore[id];
            Ne[id][is] = Nestore[id];
            Tm[id][is] = Tmstore[id];
            Te[id][is] = Testore[id];
        }
        x0meas[is][0] = X0st[0]; 
        y0meas[is][0] = Y0st[0]; 
        thmeas[is][0] = Thst[0];
        phmeas[is][0] = Phst[0];
        Emeas[is][0]  = Est[0];
        x0meas[is][1] = X0st[1]; 
        y0meas[is][1] = Y0st[1]; 
        thmeas[is][1] = Thst[1];
        phmeas[is][1] = Phst[1];
        Emeas[is][1]  = Est[1];
    } 
    if (sigmaT2_sample!=sigmaT2_sample) {
        cout << "Warning, sigmaT2sample nan" << endl;
    }
    if (sigmaT2_sample<=0.) sigmaT2_sample = pow(4.*Nunits,2.); // 4 measurements for Nunit detectors -> dchi2 = 4N per each L -> 2N + 2N
    
    // The rest is needed to compute an approximated formula for the variance of the LRT, and dsigmalrt_dx, dy values
    // --------------------------------------------------------------------------------------------------------------
    double emg   = Emeas[is][0];
    double emp   = Emeas[is][1];
    double x0g   = x0meas[is][0];
    double x0p   = x0meas[is][1];
    double y0g   = y0meas[is][0];
    double y0p   = y0meas[is][1];
    double thmg  = thmeas[is][0];
    double thmp  = thmeas[is][1];
    double phmg  = phmeas[is][0];
    double phmp  = phmeas[is][1];
    double et    = TrueE[is];
    double xt    = TrueX0[is];
    double yt    = TrueY0[is];
    double tt    = TrueTheta[is];
    double tp    = TruePhi[is];

    // This has been commented out for now:
    // If the shower can't be reconstructed reasonably for either hypothesis, we remove it from consideration
    // We set the threshold at p=10^-3 per detector and particle type. This makes the threshold pvalue be ptot < 10^(-6*Nunits)
    // double logpthresh = -6.*Nunits*log_10;
    /*if (logLG<logpthresh && logLP<logpthresh) {
        Active[is] = false;
        sigmaLRT[is] = 100.; // or whatever
        return;
    }*/

    // If we write 
    //    LLR = sum_i {-lambda_mug + lambda_mup + N_mu [log(lambda_mug)-log(lambda_mup)] +
    //                 -0.5[(T_mu-T_expg)^2-(T_mu-T_expp)^2]/sigma_t^2 + (e terms)}
    // we can differentiate directly wrt lambdas and T_exps, obtaining
    //    s^2_LLR = sum_i [ (dLLR/dlambda_mug)^2 sigma^2(lambda_mug) + 
    //                      (dLLR/dlambda_mup)^2 sigma^2(lambda_mup) +
    //                      (dLLR/dNmu)^2 sigma^2 Nmu +                              <--- check if correct to add this. NOTE IT USES TRUE R
    //                      (dLLR/dTexpg)^2 sigma_T_expg^2 + 
    //                      (dLLR/dTexpp)^2 sigma_T_expp^2 + 
    //                      (dLLR/dTmu)^2 sigma_Tmeas^2 +                            <--- same
    //                      (e terms) ]
    // This becomes, as sigma^2(lambda) is -[d^2logL/dlambda^2]^-1 = lambda^2/N,
    //    s^2_LLR = sum_i { [(N_mu-lambda_mug)^2 + (N_mu-lambda_mup)^2] / N_mu + 
    //                      [(N_e-lambda_eg)^2 + (N_e-lambda_ep)^2] / N_e +          
    //                      [log(lambda_mug)-log(lambda_mup)]^2 * N_mu +             <--- check if correct to add this
    //                      [log(lambda_eg)-log(lambda_ep)]^2 * N_e +                <--- check if correct to add this
    //                      [(T_mu+T_e-2T_expg)^2 + (T_mu+T_e-2T_expp)^2] / sigma_texp^2 +
    //                      [-(T_mu+T_e-2T_expg)^2 -(T_mu+T_e-2T_expp)^2] / sigma_tmeas^2 }   <--- same
    //
    // Then, the derivative with respect to x[id] is found as follows:
    //  ds^2/dxi = [ -2(Nm-lmg)/Nm * dlmg/dRg * dRg/dxi -2(Nm-lmp)/Nm * dlmp/dRp * dRp/dxi +
    //               -2(Ne-leg)/Ne * dleg/drg * dRg/dxi -2(Ne-lep)/Ne * dlep/dRp * dRp/dxi +
    //               2Nm*(log(lmg)-log(lmp))*(1/lmg * dlmg/dRg * dRg/dxi + 1/lmp * dlmp/dRp * dRp/dxi) +
    //               2Ne*(log(leg)-log(lep))*(1/leg * dleg/dRg * dRg/dxi + 1/lep * dlep/dRp * dRp/dxi)] 
    //
    // and for now there are no contributions from time derivatives as they cancel, given that we have set equal sigma_time and sigma_texp.         
    // ------------------------------------------------------------------------------------------------------------------------------------
    for (int id=0; id<Nunits; id++) {        
        double xi = x[id];
        double yi = y[id];
        float nm  = Nm[id][is];
        float ne  = Ne[id][is];
        double thisRg = EffectiveDistance (xi,yi,x0g,y0g,thmg,phmg,0);
        double ctg    = cos(thmg);
        double lmg    = MFromG (emg, thmg, thisRg, 0) * ctg;
        double leg    = EFromG (emg, thmg, thisRg, 0) * ctg;        
        if (nm>0.) sigmaT2_approx += pow(nm-lmg,2)/nm;
        if (ne>0.) sigmaT2_approx += pow(ne-leg,2)/ne;
        double thisRp = EffectiveDistance (xi,yi,x0p,y0p,thmp,phmp,0);
        double ctp    = cos(thmp);
        double lmp    = MFromP (emp, thmp, thisRp, 0) * ctp;
        double lep    = EFromP (emp, thmp, thisRp, 0) * ctp;
        if (nm>0.) sigmaT2_approx += pow(nm-lmp,2)/nm;
        if (ne>0.) sigmaT2_approx += pow(ne-lep,2)/ne;

        // Take in the contribution from the variation of nm, ne. This requires us to compute
        // lambdas using the true shower parameters, not the ones that maximize L!
        // ----------------------------------------------------------------------------------
        double R    = EffectiveDistance (xi,yi,xt,yt,tt,tp,0);
        double ctt  = cos(tt);
        double lmgt = MFromG (et, tt, R, 0) * ctt;
        double legt = EFromG (et, tt, R, 0) * ctt;
        double lmpt = MFromP (et, tt, R, 0) * ctt;
        double lept = EFromP (et, tt, R, 0) * ctt;
        if (lmgt*lmpt>0.) sigmaT2_approx += pow(log(lmgt)-log(lmpt),2.)*nm;
        if (legt*lept>0.) sigmaT2_approx += pow(log(legt)-log(lept),2.)*ne;

        // Add contribution from time variation. This is commented out since
        // we have taken the two sigmas to be equal for now. 
        // -----------------------------------------------------------------
        /*
        double tm = Tm[id][is];
        double te = Te[id][is];
        double thisTg   = EffectiveTime(x[id],y[id],x00,y00,thm0,phm0,0);
        double thisTp   = EffectiveTime(x[id],y[id],x01,y01,thm1,phm1,0);
        sigmaT2_approx += pow(tm+te-2*thisTg,2)/sigma2_time;
        sigmaT2_approx += pow(tm+te-2*thisTp,2)/sigma2_time;
        sigmaT2_approx += -pow(tm+te-2*thisTg,2)/sigma2_texp; // verify where else this applies instead of sigma2_time!
        sigmaT2_approx += -pow(tm+te-2*thisTp,2)/sigma2_texp;
        */

        // Compute derivative of sigmaLRT^2 over dxi, dyi
        // ----------------------------------------------
        double dlmg_drg = MFromG (emg,thmg,thisRg,1) * ctg;
        double dlmp_drp = MFromP (emp,thmp,thisRp,1) * ctp;
        double dleg_drg = EFromG (emg,thmg,thisRg,1) * ctg;
        double dlep_drp = EFromP (emp,thmp,thisRp,1) * ctp;
        // And we need derivatives wrt R with true shower params, too
        // ----------------------------------------------------------
        double dlmg_dr  = MFromG (et,tt,R,1) * ctt;
        double dlmp_dr  = MFromP (et,tt,R,1) * ctt;
        double dleg_dr  = EFromG (et,tt,R,1) * ctt;
        double dlep_dr  = EFromP (et,tt,R,1) * ctt;
        // And finally derivatives of radius true and measured vs x and y
        // --------------------------------------------------------------
        double dr_dxi   = -EffectiveDistance (xi,yi,xt,yt,tt,tp,1); // Note minus sign, as mode 1 is derivative wrt x0
        double dr_dyi   = -EffectiveDistance (xi,yi,xt,yt,tt,tp,2); // same, mode 2, y0
        double drg_dxi  = -EffectiveDistance (xi,yi,x0g,y0g,thmg,phmg,1);
        double drp_dxi  = -EffectiveDistance (xi,yi,x0p,y0p,thmp,phmp,1);
        double drg_dyi  = -EffectiveDistance (xi,yi,x0g,y0g,thmg,phmg,2);
        double drp_dyi  = -EffectiveDistance (xi,yi,x0p,y0p,thmp,phmp,2);

        double dsdx = 0.;
        double dsdy = 0.;
        if (nm>0.) {
            dsdx += -2.*(nm-lmg)/nm * dlmg_drg * drg_dxi 
                    -2.*(nm-lmp)/nm * dlmp_drp * drp_dxi; 
            dsdy += -2.*(nm-lmg)/nm * dlmg_drg * drg_dyi 
                    -2.*(nm-lmp)/nm * dlmp_drp * drp_dyi; 
            if (lmg*lmp>0.) { // note, here we need the derivatives wrt the true R!
                double factor = 2.*nm*(log(lmg)-log(lmp))*(1./lmg * dlmg_dr + 1./lmp * dlmp_dr);
                dsdx += factor * dr_dxi;
                dsdy += factor * dr_dyi;
            } 
        }
        if (ne>0.) {
            dsdx += -2.*(ne-leg)/ne * dleg_drg * drg_dxi 
                    -2.*(ne-lep)/ne * dlep_drp * drp_dxi;
            dsdy += -2.*(ne-leg)/ne * dleg_drg * drg_dyi 
                    -2.*(ne-lep)/ne * dlep_drp * drp_dyi;
            if (leg*lep>0.) { // note, here we need the derivatives wrt the true R!
                double factor = 2.*ne*(log(leg)-log(lep))*(1./leg * dleg_dr + 1./lep * dlep_dr);
                dsdx += factor * dr_dxi;
                dsdy += factor * dr_dyi;
            }
        }
        dsigma2_dx[id][is] = dsdx;
        dsigma2_dy[id][is] = dsdy;
    }
    if (sigmaT2_approx!=sigmaT2_approx) {
        cout << "Warning, sigmaT2 nan" << endl;
    }
    if (sigmaT2_approx<=0.) sigmaT2_approx = pow(4.*Nunits,2.);
    if (SampleT) {
        sigmaLRT[is] = sqrt(sigmaT2_sample);
    } else {
        sigmaLRT[is] = sqrt(sigmaT2_approx);
    }
    if (SampleT) {
        SvsS->Fill (log(sqrt(sigmaT2_approx)),log(sqrt(sigmaT2_sample)));
        SvsSP->Fill (log(sqrt(sigmaT2_approx)),log(sqrt(sigmaT2_sample)));
    }
    return;
}

// Calculation of dlogLR over dx, dy
// ---------------------------------
std::pair<double,double> dlogLR_dxy (int id, int is) { 

    // We need to find a variation of T = logLg^max-logLp^max over the distance of shower is 
    // from a detector id. T varies if we change the distances, as expected fluxes vary with measured
    // distances. But T varies also because if we change the true distance from the shower
    // center the observed fluxes of particles change.
    // ----------------------------------------------------------------------------------------------
    double xi  = x[id];
    double yi  = y[id];
    double xmg = x0meas[is][0];
    double ymg = y0meas[is][0];
    double tmg = thmeas[is][0];
    double pmg = phmeas[is][0];
    double emg = Emeas[is][0];
    double xmp = x0meas[is][1];
    double ymp = y0meas[is][1];
    double tmp = thmeas[is][1];
    double pmp = phmeas[is][1];
    double emp = Emeas[is][1];
    double xt  = TrueX0[is];
    double yt  = TrueY0[is];
    double tt  = TrueTheta[is];
    double pt  = TruePhi[is];
    double et  = TrueE[is];
    double thisRg =  EffectiveDistance (xi,yi,xmg,ymg,tmg,pmg,0);
    double thisRp =  EffectiveDistance (xi,yi,xmp,ymp,tmp,pmp,0);
    double thisRt =  EffectiveDistance (xi,yi,xt,yt,tt,pt,0);
    double dRgdx  = -EffectiveDistance (xi,yi,xmg,ymg,tmg,pmg,1);
    double dRpdx  = -EffectiveDistance (xi,yi,xmp,ymp,tmp,pmp,1);
    double dRtdx  = -EffectiveDistance (xi,yi,xt,yt,tt,pt,1);
    double dRgdy  = -EffectiveDistance (xi,yi,xmg,ymg,tmg,pmg,2);
    double dRpdy  = -EffectiveDistance (xi,yi,xmp,ymp,tmp,pmp,2);
    double dRtdy  = -EffectiveDistance (xi,yi,xt,yt,tt,pt,2);
    double thisTg = 0.;
    double thisTp = 0.;
    if (!OrthoShowers) { 
        thisTg = EffectiveTime (xi,yi,xmg,ymg,tmg,pmg,0);
        thisTp = EffectiveTime (xi,yi,xmp,ymp,tmp,pmp,0);
    }
    double dx0 = xi - xmg;
    double dy0 = yi - ymg;
    double dx1 = xi - xmp;
    double dy1 = yi - ymp;
    double ct0 = cos(tmg);
    double ct1 = cos(tmp);
    double st0 = sin(tmg);
    double st1 = sin(tmp);
    double cp0 = cos(pmg);
    double cp1 = cos(pmp);
    double sp0 = sin(pmg);
    double sp1 = sin(pmp);
    double lambdaMG0, lambdaEG0;
    double lambdaMP0, lambdaEP0;
    double lambdaMG, lambdaEG;
    double lambdaMP, lambdaEP;

    // The calculation goes as follows, for an event k and detector unit i:
    // logLG = {-lambda_mug_i - lambda_eg_i + N_mug_i*log(lambda_mug_i) +
    //         N_eg_i*log(lambda_eg_i) -log(N_mu_i!) - log(N_e_i!) }
    // from which we get:
    // dlogLG/dR = { -dlambda_mug_i/dR_i - dlambda_eg_i/dR_i +d/dR_i(N_mug_i*log(lambda_mug_i)) +
    //             d/dR_i(N_eg_i*log(lambda_eg_i)) + d/dR_i(-log(N_mu_i!)) + d_dR_i(-log(N_mu_i!)) }
    // dlogLP/dR = { -dlambda_mup_i/dR_i - dlambda_ep_i/dR_i +d/dR_i(N_mup_i*log(lambda_mup_i)) +
    //             d/dR_i(N_ep_i*log(lambda_ep_i)) + d/dR_i(-log(N_mup_i!)) + d_dR_i(-log(N_mup_i!)) }
    //
    // Now, the factorials are derivatives with respect to true R, so they cancel in the logLR and we
    // ignore them. Instead, while the R_i deriving the lambdas are measured ones, the R_i deriving 
    // the N_ are the true ones. For these, we substitute N_ with the expectation values lambda at the 
    // same location. Further, since some of the factors depend on Rmeas and others on Rtrue, we find
    // it better to derive directly with respect to x[id], y[id].
    // We finally get the following expression:
    //    dlogLG/dx = dlambdag_mu/dRg_meas * dRg_meas/dx * (N_mu/lambdag_mu - 1) + 
    //                dlambdag_mu/dR_true dR_true/dx log(lambdag_mu) +             (note N->lambda here)
    //                dlambdag_e/dRg_meas * dRg_meas/dx * (N_e/lambdag_e - 1) +
    //                dlambdag_e/dR_true dR_true/dx log(lambdag_e)                 (and here)
    // and similarly for the protons.
    // As far as time dependence goes, we note that Tmeas changes with Rtrue, and Texp with Rmeas,
    // so we need to account for both and 
    //    dlogLG/dx += -(Tmeas-Texp)/sigma^2 * dTmeas/dRtrue * dRtrue/dx +
    //                 +(Tmeas-Texp)/sigma^2 * dTexp/dRmeas  * dRmeas/dx      
    // -----------------------------------------------------------------------------------------------

    // These are fluxes from measured shower coords
    // --------------------------------------------
    lambdaMG0 = MFromG (emg, tmg, thisRg, 0);
    lambdaEG0 = EFromG (emg, tmg, thisRg, 0);
    lambdaMP0 = MFromP (emp, tmp, thisRp, 0);
    lambdaEP0 = EFromP (emp, tmp, thisRp, 0);
    lambdaMG = lambdaMG0 * ct0;
    lambdaEG = lambdaEG0 * ct0;
    lambdaMP = lambdaMP0 * ct1;
    lambdaEP = lambdaEP0 * ct1;

    // Derivatives with respect to Rmeas
    // ---------------------------------
    double dlMGdR, dlEGdR;
    double dlMPdR, dlEPdR;
    dlMGdR = MFromG (emg, tmg, thisRg, 1) * ct0;
    dlEGdR = EFromG (emg, tmg, thisRg, 1) * ct0;
    dlMPdR = MFromP (emp, tmp, thisRp, 1) * ct1;
    dlEPdR = EFromP (emp, tmp, thisRp, 1) * ct1;

    // Now assemble the pieces
    // -----------------------
    double dlogLGdR = -dlMGdR-dlEGdR; // this is the contribution of dlambda/dRmeas values
    double dlogLPdR = -dlMPdR-dlEPdR;
    float nm       = Nm[id][is];
    float ne       = Ne[id][is];
    float tm       = Tm[id][is];
    float te       = Te[id][is];
    if (lambdaMG>0.) {
        dlogLGdR += nm/lambdaMG * dlMGdR; 
    }
    if (lambdaEG>0.) {
        dlogLGdR += ne/lambdaEG * dlEGdR;
    }
    if (lambdaMP>0.) {
        dlogLPdR += nm/lambdaMP * dlMPdR; 
    }
    if (lambdaEP>0.) {
        dlogLPdR += ne/lambdaEP * dlEPdR; 
    }

    // Now the result of above sums has to be multiplied by dR_meas/dx or dy
    // ---------------------------------------------------------------------
    double dlogLGdx, dlogLGdy, dlogLPdx, dlogLPdy;
    dlogLGdx = dlogLGdR * dRgdx;
    dlogLPdx = dlogLPdR * dRpdx; 
    dlogLGdy = dlogLGdR * dRgdy;
    dlogLPdy = dlogLPdR * dRpdy;

    // We now add the contribution log(lambda_mu) * dN_mu/dR_true * dR_true/dx (dy), and e term.
    // Here we take N_mu == lambda_mu to get its R_true derivative.
    // -----------------------------------------------------------------------------------------
    double factor = cos(tt)*dRtdx;
    if (lambdaMG>0.) dlogLGdx += MFromG (et,tt,thisRt,1) * factor * log(lambdaMG); 
    if (lambdaEG>0.) dlogLGdx += EFromG (et,tt,thisRt,1) * factor * log(lambdaEG); 
    if (lambdaMP>0.) dlogLPdx += MFromP (et,tt,thisRt,1) * factor * log(lambdaMP); 
    if (lambdaEP>0.) dlogLPdx += EFromP (et,tt,thisRt,1) * factor * log(lambdaEP); 
    factor = cos(tt)*dRtdy;
    if (lambdaMG>0.) dlogLGdy += MFromG (et,tt,thisRt,1) * factor * log(lambdaMG); 
    if (lambdaEG>0.) dlogLGdy += EFromG (et,tt,thisRt,1) * factor * log(lambdaEG); 
    if (lambdaMP>0.) dlogLPdy += MFromP (et,tt,thisRt,1) * factor * log(lambdaMP); 
    if (lambdaEP>0.) dlogLPdy += EFromP (et,tt,thisRt,1) * factor * log(lambdaEP); 

    // Handle variation of logL on R due to variation of thisT
    // -------------------------------------------------------
    if (!OrthoShowers) { 
        double t0      = dx0*st0*cp0 + dy0*st0*sp0;
        double t1      = dx1*st1*cp1 + dy1*st1*sp1;
        double den01   = dx0-t0*st0*cp0;
        double den02   = dy0-t0*st0*sp0;
        double den11   = dx1-t1*st1*cp1;
        double den12   = dy1-t1*st1*sp1;
        double dTg_dRi = maxdTdR;
        if (den01!=0. && den02!=0.) dTg_dRi = thisRg/c0 * ( st0*cp0/den01 + st0*sp0/den02 );
        double dTp_dRi = maxdTdR;
        if (den11!=0. && den12!=0.) dTp_dRi = thisRp/c0 * ( st1*cp1/den11 + st1*sp1/den12 );
        double dTtrue_dxtrue = EffectiveTime (xi,yi,xt,yt,tt,pt,1);
        double dTg_dxmeas    = EffectiveTime (xi,yi,xmg,ymg,tmg,pmg,1);
        double dTp_dxmeas    = EffectiveTime (xi,yi,xmp,ymp,tmp,pmp,1);
        if (nm>0) {
            dlogLGdx +=  (tm-thisTg)/sigma2_time * dTg_dxmeas;
            dlogLPdx +=  (tm-thisTp)/sigma2_time * dTp_dxmeas;
            dlogLGdx += -(tm-thisTg)/sigma2_time * dTtrue_dxtrue;
            dlogLPdx += -(tm-thisTp)/sigma2_time * dTtrue_dxtrue;
        } 
        if (ne>0) {
            dlogLGdx +=  (te-thisTg)/sigma2_time * dTg_dxmeas;
            dlogLPdx +=  (te-thisTp)/sigma2_time * dTp_dxmeas;
            dlogLGdx += -(te-thisTg)/sigma2_time * dTtrue_dxtrue;
            dlogLPdx += -(te-thisTp)/sigma2_time * dTtrue_dxtrue;
        }
        double dTtrue_dytrue = EffectiveTime (xi,yi,xt,yt,tt,pt,2);
        double dTg_dymeas    = EffectiveTime (xi,yi,xmg,ymg,tmg,pmg,2);
        double dTp_dymeas    = EffectiveTime (xi,yi,xmp,ymp,tmp,pmp,2);
        if (nm>0) {
            dlogLGdy +=  (tm-thisTg)/sigma2_time * dTg_dymeas;
            dlogLPdy +=  (tm-thisTp)/sigma2_time * dTp_dymeas;
            dlogLGdy += -(tm-thisTg)/sigma2_time * dTtrue_dytrue;
            dlogLPdy += -(tm-thisTp)/sigma2_time * dTtrue_dytrue;
        }
        if (ne>0) {
            dlogLGdy +=  (te-thisTg)/sigma2_time * dTg_dymeas;
            dlogLPdy +=  (te-thisTp)/sigma2_time * dTp_dymeas;
            dlogLGdy += -(te-thisTg)/sigma2_time * dTtrue_dytrue;
            dlogLPdy += -(te-thisTp)/sigma2_time * dTtrue_dytrue;
        }
    } // end if !OrthoShowers

    if (dlogLGdR!=dlogLGdR || dlogLPdR!=dlogLPdR) {
        cout << "      Warning - Trouble in dlogLR_dR" << endl;
        cout << thisTg << " " << thisTp << " " << sigma2_time << endl; //  << " " << dTg_dRi << " " << dTp_dRi << endl;
        cout << thisRg << " " << thisRp << " " << dlMGdR << " " << dlEGdR << " " << dlMPdR << " " << dlEPdR << endl;
        cout << lambdaMG << " " << lambdaEG << " " << lambdaMP << " " << lambdaEP << endl << endl; 
        SaveLayout();
        warnings3++;
        return std::make_pair(0.,0.);
    }
    // Return function value
    // ---------------------
    //cout << "id, is = " << id << " " << is << " " << dlogLGdx-dlogLPdx << " " << dlogLGdy-dlogLPdy << endl;
    return std::make_pair(dlogLGdx-dlogLPdx,dlogLGdy-dlogLPdy);
}

// This routine computes the PDF of the test statistic for a primary hypothesis
// ----------------------------------------------------------------------------
double ComputePDF (int k, bool gammahyp) {

    // We introduce the calculation of the probability that at least N>=Ntrigger
    // detectors observe >0 particles (muons plus (ele+gamma)) and incorporate it
    // into the probability that the shower is a gamma or a proton, as the event
    // will be in the considered sample only if N>=Ntrigger units saw a signal.
    // This is computed by taking a Poisson approximation, as follows:
    // Each detector has an expectation value for the number of observed particles
    // that is equal to 
    //     xi_i = lambda_mu(i)+lambda_e(i). 
    // This depends of course on the hypothesis for the considered shower. We have
    //     P_i(>=1 particle) = 1 - exp(-xi_i)
    // and we form the sum of these P_i for all detectors:
    //     S = sum_i P_i
    // We then get the probability that the event passes the trigger, and is thus
    // included in the sample, as the approximated value
    //     Pactive[is] = 1 - sum_{k=0}^{Ntrigger-1} Poisson(k|S)
    // This is checked to be a good representation of the true probability by the routine
    // at the end of this macro (CheckProb).
    // ----------------------------------------------------------------------------------
    double p     = 0.;
    double Norm  = 0.;
    for (int m=0; m<Nevents; m++) {
        if (!Active[m] || IsGamma[m] !=gammahyp) continue;
        double sigma = sigmaLRT[m];
        double Gden  = sqrt2pi*sigma; 
        double G     = PActive[m]/Gden*exp(-pow((logLRT[m]-logLRT[k])/sigma,2.)/2.);
        if (G!=G) {
            cout << "Warning, NANs in PDF calculations " << endl;
            warnings4++;
            continue; 
        }
        if (G>epsilon2) {
            p += G; // otherwise it screws around with E-170 numbers and returns bogus results
        }
        Norm += PActive[m]; //also zeros contribute to denominator        
    }
    // We have summed Gaussians (already normalized) an effective number of times equal to Ng or Np,
    // so to normalize the pdfs we need to divide by those numbers. This includes PActive[k] terms.
    // --------------------------------------------------------------------------------------------- 
    if (Norm>0.) p /= Norm; 
    if (p<epsilon) p = epsilon; // Protect against outliers screwing up calculations
    return p;
}

// Compute variation of dlogL_dE for small variation of E. This is used to extract
// an estimate of dEk/dRik for shower k and detector i, in the calculation of the
// derivative of the integrated resolution, for the Utility calculation.
// Because it is too CPU costly to re-maximize the logLikelihood for a variation
// of detector position, we compute delta(dlogL/dE)/deltaE and delta(dlogL/dE)/deltaR,
// and force them to be equal and of opposite sign. The routine below computes the
// first part. It is called by the other routine, which determines dE_dR altogether.
// 
// Note, here we are only concerned with energy estimates under the gamma hypothesis!
// ---------------------------------------------------------------------------------------------
double delta_dlogLdE (int is, double de) {
    double ethis        = Emeas[is][0];
    double eprime       = ethis + de;
    double dlogLdE      = 0.;
    double dlogLdEprime = 0.;
    double ct           = cos(thmeas[is][0]);
    for (int id=0; id<Nunits; id++) {
        float nm        = Nm[id][is];
        float ne        = Ne[id][is];
        double thisR    = EffectiveDistance (x[id],y[id],x0meas[is][0],y0meas[is][0],thmeas[is][0],phmeas[is][0],0);          
        double lambdaM0 = MFromG (ethis, thmeas[is][0],thisR, 0);
        double lambdaE0 = EFromG (ethis, thmeas[is][0],thisR, 0);
        double dlM0dE   = MFromG (ethis, thmeas[is][0],thisR, 2);
        double dlE0dE   = EFromG (ethis, thmeas[is][0],thisR, 2);
        dlogLdE        -= (dlM0dE + dlE0dE)*ct;
        if (lambdaM0>0.) {
            dlogLdE += nm/lambdaM0 * dlM0dE; 
        }
        if (lambdaE0>0.)  {
            dlogLdE += ne/lambdaE0 * dlE0dE;
        }
        // Now for modified energy, same detector positions
        lambdaM0        = MFromG (eprime, thmeas[is][0],thisR, 0);
        lambdaE0        = EFromG (eprime, thmeas[is][0],thisR, 0);
        dlM0dE          = MFromG (eprime, thmeas[is][0],thisR, 2);
        dlE0dE          = EFromG (eprime, thmeas[is][0],thisR, 2);
        dlogLdEprime   -= (dlM0dE + dlE0dE)*ct;
        if (lambdaM0>0.) {
            dlogLdEprime += nm/lambdaM0 * dlM0dE; 
        }
        if (lambdaE0>0.)  {
            dlogLdEprime += ne/lambdaE0 * dlE0dE;
        }
    }
    // cout << " E reco: dlogLdE = " << dlogLdE << " Em,Et = " << Emeas[is][0] << "," << TrueE[is] << endl;
    return dlogLdEprime - dlogLdE; 
}

// Compute derivative of energy over R (shower, detector) by procedure explained above 
// (See routine delta_dlogLdE)
// -----------------------------------------------------------------------------------                     
double dEk_dRik (int id, int is) {

    // Compute variation in dlogL/dE for tiny E change
    // -----------------------------------------------
    double emeas     = Emeas[is][0];
    double tmeas     = thmeas[is][0];
    double de = 0.0001*emeas;
    if (emeas==10.) de = -de; // Only consider downward shifts if E is at max boundary
    double deltaf_E = delta_dlogLdE(is,de); 

    // Now we search for the dR that produces the opposite shift in dlogLdE
    // --------------------------------------------------------------------
    float nm         = Nm[id][is];
    float ne         = Ne[id][is];
    double thisR     = EffectiveDistance (x[id],y[id],x0meas[is][0],y0meas[is][0],tmeas,phmeas[is][0],0);
    double ct        = cos(thmeas[is][0]);
    double lambdaM0  = MFromG (emeas, tmeas,thisR, 0);
    double lambdaE0  = EFromG (emeas, tmeas,thisR, 0);
    double dlM0dE    = MFromG (emeas, tmeas,thisR, 2);
    double dlE0dE    = EFromG (emeas, tmeas,thisR, 2);
    double dlogLdE   = -(dlM0dE + dlE0dE)*ct;
    if (lambdaM0>0.) {
        dlogLdE += nm/lambdaM0 * dlM0dE; 
    }
    if (lambdaE0>0.)  {
        dlogLdE += ne/lambdaE0 * dlE0dE;
    }
    int iloop           = 0;
    int maxloops        = 20;
    double distmax      = 0.000000001;
    double dist         = 0.;
    double olddist      = largenumber;
    double maxincrement = 2.*spanR;
    double increment    = 0.001*DetectorSpacing;
    double oldincrement = 0.;
    double thisRprime   = thisR + increment;
    double dlogLdEprime = 0.;
    do {
        if (thisRprime<Rmin) thisRprime = Rmin; // protect it
        lambdaM0            = MFromG (emeas, tmeas, thisRprime, 0);
        lambdaE0            = EFromG (emeas, tmeas, thisRprime, 0);
        dlM0dE              = MFromG (emeas, tmeas, thisRprime, 2);
        dlE0dE              = EFromG (emeas, tmeas, thisRprime, 2);
        dlogLdEprime        = -(dlM0dE + dlE0dE)*ct;
        if (lambdaM0>0.) {
            dlogLdEprime += nm/lambdaM0 * dlM0dE; 
        }
        if (lambdaE0>0.)  {
            dlogLdEprime += ne/lambdaE0 * dlE0dE;
        }
        dist = deltaf_E - (dlogLdEprime-dlogLdE);
        // cout << " il = " << iloop << " dist = " << dist << "old dist = " << olddist << " incr, old = " << increment << " " << oldincrement << endl;
        if (fabs(dist)<fabs(distmax)) continue; // we've done it
        if (iloop>0) {
            double R = fabs(dist)/fabs(olddist);
            if (dist*olddist>0.) { // on same side of zero
                if (R<1.) { // we reduced it in size
                    double tmp = oldincrement;
                    oldincrement = increment;
                    increment += R/(1.-R)*(increment-tmp); // linear approximation for increment
                    if (fabs(increment)>maxincrement) {
                        if (increment>0) {
                            increment = maxincrement;
                        } else {
                            increment = -maxincrement;
                        }
                    }
                } else if (R>1.) { // we increased it, go back
                    double tmp = oldincrement;
                    oldincrement = increment;
                    increment = tmp +0.5*(tmp-increment); // jump on the other side
                 } else {
                    oldincrement = increment;
                    increment *= 1.2;
                }
            } else if (dist*olddist<0.) {
                if (R<1.) {
                    double tmp = oldincrement;
                    oldincrement = increment;
                    increment = increment +1./(1.+1./R)*(tmp-increment); // linear approx
                    if (fabs(increment)>maxincrement) {
                        if (increment>0) {
                            increment = maxincrement;
                        } else {
                            increment = -maxincrement;
                        }
                    }
                } else if (fabs(dist)>fabs(olddist)) {
                    double tmp = increment;
                    oldincrement = increment;
                    increment = tmp + 0.5*(tmp-increment); // jump on other side
                } else {
                    oldincrement = increment;
                    increment *= 0.8;
                }
            }
        } else {
            oldincrement = increment;
            increment = 2.*increment;
        }
        thisRprime = thisR + increment;
        if (thisRprime<0.) thisRprime =0.;
        olddist = dist;
        iloop++;
    } while (fabs(dist)>fabs(distmax) && iloop<maxloops && increment<4.*spanR);
    //cout << "     N loops = " << iloop << " dist = " << dist << " E, Et = " << Emeas[is][0] << " " << TrueE[is] 
    //     << " increment = " << increment << endl;
    //if (fabs(dist)>1000.) cout << "df = " << deltaf_E << " dlogLdeprime = " << dlogLdEprime << " dlogLde = " << dlogLdE << endl;

    if (fabs(dist)<fabs(distmax)) {
    //    if (fabs(de/increment)>10.) {
    //        cout << " de, incr = " << de << " " << increment << " dist = " << dist << " xd,yd = " << x[id] << "," << y[id] 
    //             << " x0,y0m = " << x0meas[is][0] << " " << y0meas[is][0] << " Em = " << Emeas[is][0] << endl; 
    //    }
        return de/increment; 
    } else {
        return 0.; // failure
    }
}

// Compute utility function, gamma fraction part
// ---------------------------------------------
double ComputeUtilityGF () {
    double U_GF = 0.;
    cout    << "     GF = " << MeasFg << " +- " << MeasFgErr << " (true = " << TrueGammaFraction << ")"; 
    outfile << "     GF = " << MeasFg << " +- " << MeasFgErr << " (true = " << TrueGammaFraction << ")"; 
    U_GF = eta_GF * MeasFg/MeasFgErr;
    U_GF = U_GF * ExposureFactor;
    cout    << " U_GF = " << U_GF;
    outfile << " U_GF = " << U_GF;
    return U_GF;
}

// Compute utility function and gradient, integrated resolution version
// --------------------------------------------------------------------
double ComputeUtilityIR () {
    double U_IR = 0.;
    U_IR_Num = 0.;
    U_IR_Den = 0.;
    for (int is=Nevents; is<Nevents+Nbatch; is++) {
        if (IsGamma[is] && Active[is]) {
            // We compute the U_IR piece considering that each batch event contributes to the estimate of the
            // integrated resolution with a weight (1+wsl*log())*PActive, where PActive accounts for the 
            // probability of that event to have made it to the set. When differentiating wrt xi, we then
            // include the contribution of dPActive/dx,dy
            // ----------------------------------------------------------------------------------------------
            double weight = PActive[is] * (1.+Wslope*log(TrueE[is]/Emin)); 
            double Et  = TrueE[is];
            double Et2 = Et*Et;
            double dE2 = pow(Et-Emeas[is][0],2);
            U_IR_Num += weight * Et2/(dE2+Et2*delta2);
            U_IR_Den += weight;
        }
    }
    if (U_IR_Den!=0.) U_IR = eta_IR * U_IR_Num / U_IR_Den;
    cout    << " U_IR = " << U_IR;
    outfile << " U_IR = " << U_IR;
    return U_IR;
}

// Function to be executed by each thread to generate and reconstruct events
// -------------------------------------------------------------------------
void threadFunction (int threadId) {

    int ismin = threadId*(Nevents+Nbatch)/Nthreads;
    int ismax = (threadId+1)*(Nevents+Nbatch)/Nthreads;
    if (threadId == Nthreads-1) ismax = Nevents+Nbatch; // Protect against rounding of integer divisions

    // Lock the mutex before accessing the shared data
    // std::lock_guard<std::mutex> lock(dataMutex);

    // Perform operations on the shared data


    // We compute the templates of test statistic 
    // (log-likelihood ratio) for gammas and protons, and while
    // we are at it, we also generate extra Nbatch events for SGD
    // ----------------------------------------------------------
    for (int is=ismin; is<ismax; is++) {
        if (Nthreads==1 && (is+1)%(Nevents/4)==0) {
            cout << is+1 << " ";
            outfile << is+1 << " ";
        }

        // Zero LRT and sigmaLRT arrays
        // ----------------------------
        logLRT[is]   = 0.;
        sigmaLRT[is] = 1.;

        // Find Nm[], Ne[] for this event
        // -------------------------------
        if (is%2==0) { // Generate a gamma 
            IsGamma[is] = true;
        } else { // Generate a proton 
            IsGamma[is] = false;
        }
        GenerateShower (is);
        if (!Active[is]) continue;

        // For this shower, find value of test statistic. NB this requires that we have truex0, truey0,
        // and the mug, eg, mup, ep values filled for this event. GenerateShower takes care of the latter
        // ----------------------------------------------------------------------------------------------
        FindLogLR (is); // Fills logLRT[] array and fills sigmaLRT - NB We rely on mug,eg,mup,ep being def above               

    } // end is loop
    if (Nthreads==1) {
        cout << endl;
        outfile << endl;
    }
}

// Multithreading function computing derivatives of U wrt x,y
// Call with threadId = 0 for no threading 
// ----------------------------------------------------------
void threadFunction2 (int threadId) { 

    int idmin = threadId*Nunits/Nthreads;
    int idmax = (threadId+1)*Nunits/Nthreads;
    if (threadId == Nthreads-1) idmax = Nunits; // Protect against integer divisions

    // Lock the mutex before accessing the shared data
    // std::lock_guard<std::mutex> lock(dataMutex);

    // All calculations take place within the detector loop below
    // ----------------------------------------------------------
    for (int id=idmin; id<idmax; id++) { 

        double dldxm[maxEvents];  // This is used to store values during calculations of derivatives, to avoid multiple calculations
        double dldym[maxEvents];  // This is used to store values during calculations of derivatives, to avoid multiple calculations
        double dpg_dx[maxEvents];
        double dpp_dx[maxEvents];
        double dpg_dy[maxEvents];
        double dpp_dy[maxEvents];

        if (Nthreads==1 && (id+1)%(Nunits/10)==0) cout << id+1 << " ";
        double xi  = x[id]; // Speeds up calculations
        double yi  = y[id]; 

        if (scanU && id!=idstar) continue; // Only compute dUdx,dy for that one detector if scanning dU

        dU_dxi[id] = 0.;
        dU_dyi[id] = 0.;

        // First fill the arrays dldxm[][], dldym[][] once and for all
        // -----------------------------------------------------------
        for (int m=0; m<Nevents; m++) {
            if (!Active[m]) continue;
            std::pair<double,double> dlogLRdxy = dlogLR_dxy(id,m);
            dldxm[m] = dlogLRdxy.first;  // dlogLR_dR (id,m,1);
            dldym[m] = dlogLRdxy.second; // dlogLR_dR (id,m,2);
        }

        // New calculation of dpg_dxi, dpg_dyi, dpp_dxi, dpp_dyi, hopefully less messy
        // We have Pg,k = Sum_{m=1..Nevents,g} [PActive_m*G_{den,m}*G(Tm-Tk,s_m)] / Sum_m PActive_m
        // with 
        //      G_{den,m} = (2pi*s_m^2)^{-1/2}
        //      G(Tm-Tk,s_m) = exp(-1/2 (Tm-Tk)^2/s_m^2)
        // We get, by deriving wrt xi:
        //      dPg,k/dxi = { Sum_m [ (PA_m Gden G) * [ (Tm-Tk)/s_m^2 * (dTk/dxi-dTm/dxi) + 
        //                                              (Tm-Tk)^2/(2s_m^4) ds_m^2/dxi +
        //                                              -Gden^3 pi ds_m^2/dxi + 1/PA_m dPA_m/dxi ]] * Sum_m PA_m
        //                    - Sum_m (PA_m Gden G) * Sum_m (dPA_m/dxi) } / (Sum_m PA_m)^2
        // ------------------------------------------------------------------------------------------------------
        for (int k=Nevents; k<Nevents+Nbatch; k++) {
            if (!Active[k]) continue;
            std::pair<double,double> dlogLRdxy = dlogLR_dxy (id,k);
            double dldxk  = dlogLRdxy.first;  // dlogLR_dR (id,k,1);
            double dldyk  = dlogLRdxy.second; // dlogLR_dR (id,k,2);
            double numg = 0.;
            double deng = 0.;
            double dnum_dxg = 0.;
            double dnum_dyg = 0.;
            double dden_dxg = 0.;
            double dden_dyg = 0.;
            double nump = 0.;
            double denp = 0.;
            double dnum_dxp = 0.;
            double dnum_dyp = 0.;
            double dden_dxp = 0.;
            double dden_dyp = 0.;

            // Get components of dpg/dxi, dpg/dyi
            // ----------------------------------
            for (int m=0; m<Nevents; m++) {
                if (!Active[m]) continue;
                double Sm          = SumProbgt1[m]; 
                double sigma       = sigmaLRT[m];
                double sigma2      = sigma*sigma;
                double sigma4      = sigma2*sigma2;
                double dlmk        = logLRT[m]-logLRT[k];
                double Gden        = 1./(sqrt2pi*sigma); 
                double G           = exp(-pow(dlmk/sigma,2.)/2.); 

                // We recall that Pg is sum_m (P*Gden*G) / sum_m P
                // -----------------------------------------------
                double Factor      = PActive[m]*Gden*G;
                if (Factor!=Factor) {
                    cout << "Warning, nan in dpdx calculation " << endl;
                    cout << "Gden, dlmk, sigma, Sm, PA = " << Gden << " " << dlmk << " " << sigma << " " << Sm << " " << PActive[k] << endl;
                    warnings4++;
                    continue; 
                }

                // Get dPA_dx, dPA_dy
                // To incorporate the contribution of PActive_m, we have made it part of the def of G.
                // This means we have a term to add, G/PActive_m * dPActive_m/dxi. 
                // The latter term can be computed as follows (PActive = 1 - sum(1:Ntr-1)):
                //     dPActive_m/dxi = -d/dxi [sum_{j=0}^{Ntr-1}(e^{-Sm}*Sm^j/j!)]
                // with
                //     Sm = Sum_{i=1}^Ndet [1-exp(-lambda_mu^i-lambda_e^i)_m]
                // We get
                //     dPActive_m/dxi = Sum_{j=0}^{Ntr-1} 1/j! [e^{-Sm}*(Sm^j-j*Sm^(j-1))*dSm/dxi]
                // Now for dSm/dxi we have
                //     dSm/dxi = d/dxi [ Sum_{i=1}^{Ndet} (1-e^(-lambda_mu^i-lambda_e^i)_m))]
                //               = -d/dxi (e^[-lambda_mu^i-lambda_e^i]_m) =
                //               = e^[-lambda_mu^i-lambda_e^i]*(dlambda_mu^i/dxi + dlambda_e^i/dxi)
                // and the latter are computed in the flux routines.
                // Note that the lambdas are the true ones, as we compute PDFs with events that are
                // included in the sum if they pass the trigger, and they do if the true fluxes exceed
                // the threshold of Ntrigger detectors firing.
                // -----------------------------------------------------------------------------------
                // The calculation of the term due to dPActive/dxi is laborious, but we only need
                // to perform it if we are close to threshold, otherwise the derivative contr. is null
                // -----------------------------------------------------------------------------------
                bool consider_dP = false;
                if (fabs(Sm-Ntrigger)<1.) consider_dP = true; // 2. might be fine tuned. 1 prolly good
                if (PActive[m]>0.) { 
                    if (IsGamma[m]) {
                        dnum_dxg += (dlmk/sigma2*(dldxk-dldxm[m]) 
                                      + pow(dlmk,2.) / (2.*sigma4) * dsigma2_dx[id][m]); 
                        dnum_dxg += pi * pow(Gden,2.) * dsigma2_dx[id][m];
                        dnum_dyg += (dlmk/sigma2*(dldyk-dldym[m]) 
                                      + pow(dlmk,2.) / (2.*sigma4) * dsigma2_dy[id][m]); 
                        dnum_dyg += pi * pow(Gden,2.) * dsigma2_dy[id][m];
                        if (consider_dP) {
                            double dPAm_dx = ProbTrigger (Sm,1,id,m);
                            double dPAm_dy = ProbTrigger (Sm,2,id,m);
                            dnum_dxg += dPAm_dx / PActive[m];
                            dden_dxg += dPAm_dx;
                            dnum_dyg += dPAm_dy / PActive[m];
                            dden_dyg += dPAm_dy;
                        }
                        dnum_dxg *= Factor;
                        dnum_dyg *= Factor;
                        numg += Factor;
                        deng += PActive[m];
                    } else {
                        dnum_dxp += (dlmk/sigma2*(dldxk-dldxm[m]) 
                                      + pow(dlmk,2.) / (2.*sigma4) * dsigma2_dx[id][m]); 
                        dnum_dxp += pi * pow(Gden,2) * dsigma2_dx[id][m];
                        dnum_dyp += (dlmk/sigma2*(dldyk-dldym[m]) 
                                      + pow(dlmk,2.) / (2.*sigma4) * dsigma2_dy[id][m]); 
                        dnum_dyp += pi * pow(Gden,2) * dsigma2_dy[id][m];
                        if (consider_dP) {
                            double dPAm_dx = ProbTrigger (Sm,1,id,m);
                            double dPAm_dy = ProbTrigger (Sm,2,id,m);
                            dnum_dxp += dPAm_dx / PActive[m];
                            dden_dxp += dPAm_dx;
                            dnum_dyp += dPAm_dy / PActive[m];
                            dden_dyp += dPAm_dy;
                        }
                        dnum_dxp *= Factor;
                        dnum_dyp *= Factor;
                        nump += Factor;
                        denp += PActive[m];
                    } // end if gamma
                } // end if pactive
            } // end m loop

            // With the above sums, we can compute the x,y derivatives of PDF values for both hypotheses of event k.
            // -----------------------------------------------------------------------------------------------------
            double den2g = pow(deng,2.);
            double den2p = pow(denp,2.);
            if (den2g>0.) {
                dpg_dx[k] = (dnum_dxg*deng-numg*dden_dxg)/den2g;
                dpg_dy[k] = (dnum_dyg*deng-numg*dden_dyg)/den2g;
            }
            if (den2p>0.) {
                dpp_dx[k] = (dnum_dxp*denp-nump*dden_dxp)/den2p;
                dpp_dy[k] = (dnum_dyp*denp-nump*dden_dyp)/den2p;
            }
        } // end k loop

        // Now get variation of inverse sigma_fs over dR
        // ---------------------------------------------
        double d_invsigfs_dx  = 0.;
        double d_invsigfs_dy  = 0.;
        double d_fs_dx = 0.;
        double d_fs_dy = 0.;

        // Also deal with variation of integrated resolution on measured gamma energy
        // --------------------------------------------------------------------------
        double sumdPAkdxi    = 0.;
        double sumdPAkdyi    = 0.;
        double sumdPAkdxidE2 = 0.;
        double sumdPAkdyidE2 = 0.;
        double sum_dedx      = 0.;
        double sum_dedy      = 0.;
        double dR = 0.5*DetectorSpacing; // use a relevant distance for incremental ratio

        for (int k=Nevents; k<Nevents+Nbatch; k++) {

            // HACK! We remove showers that are too close to this detector, as they cause too much stochasticity.
            // --------------------------------------------------------------------------------------------------
            //double R = EffectiveDistance (xi,yi,TrueX0[k],TrueY0[k],TrueTheta[k],TruePhi[k],0);
            //if (R<3.*DetectorSpacing) continue;

            if (!Active[k]) continue;

            double sqrden = MeasFg * pg[k] + (1.-MeasFg)*pp[k]; 
            if (sqrden<epsilon) continue; // protect against PDF holes
            double den    = pow(sqrden,2.);
            double dif    = pg[k]-pp[k];
            double dif2   = pow(dif,2.);
            double this_dsfs_dx, this_dsfs_dy;
            if (dif!=0. && den!=0.) { // protect against adding nan to dudx
                // The contributions to d(1/sigmafs)/dx are computed by remembering that we found already
                // 1/sigmafs = sum_k [(pg_k-pp_k)^2/(MeasFg*pg_k+(1-MeasFg)*pp_k)^2]
                // --------------------------------------------------------------------------------------
                this_dsfs_dx = (dif*(dpg_dx[k]-dpp_dx[k])*den - dif2*sqrden*(MeasFg*dpg_dx[k]+(1.-MeasFg)*dpp_dx[k])) / pow(den,2);
                this_dsfs_dy = (dif*(dpg_dy[k]-dpp_dy[k])*den - dif2*sqrden*(MeasFg*dpg_dy[k]+(1.-MeasFg)*dpp_dy[k])) / pow(den,2);
                d_invsigfs_dx += this_dsfs_dx; 
                d_invsigfs_dy += this_dsfs_dy; 
                //if (fabs(this_dsfs_dx)>1.E5 | fabs(this_dsfs_dy)>1.E5) {
                //    cout << "den, dif, dpgdx, dppdx, dppdx, dppdy = " << den << " " << dif << " " << dpg_dx[k] << " " << dpg_dy[k] << " "  << dpp_dx[k] << " " << dpp_dy[k] << endl;  
                //}

                // Now compute dfs_dx, dy. We do it by using implicit differentiation.
                // We treat Fs as the dependent variable and Ps_i, Pp_i independent variables,
                // and use the chain rule to relate derivation by Ps_i, Pp_i to derivation by xi.
                // We define
                //     H = d logL / d Fs = Sum_i [(Ps_i-Pp_i)/(Fs*Ps_i+(1-Fs)*Pp_i)] = 0
                // as the equation defining Fs. We apply derivation by xk, obtaining:
                //     dH/dPs_i dPs_i/dxk + dH/dPp_i dPp_i/dxk + dH/dFs dFs/dxk = 0
                // whence
                //     dFs/dxk = -1/(dH/dFs) * sum_i { dH/dPs_i dPs_i/dxk + dH/dPp_i dPp_i/dxk }
                // Now we have
                //     dH/dFs = - Sum_j [(dif_j)^2/(den_j)^2] = -1/sigmafs^2
                // where we defined dif_j = Ps_j-Pp_j   and den_j = Fs Ps_j + (1-Fs)*Pp_j;
                //     dH/dPs_i = (den_i - dif_i*Fs)/den_i^2 = Pp_i/den_i^2
                //     dH/dPp_i = -(den_i + dif_i*(1-Fs))/den_i^2 = -Ps_i/den_i^2
                // And so
                //     dFs/dxi = { Pp_i * dPs_i/dxi - Ps_i * dPp_i/dxi } / 
                //                {(den_i)^2 * Sum_j [(dif_j)^2/(den_j)^2]}
                // Finally, note that 1./Sum_j [dif^2/den^2] = sigmafs2.
                // ------------------------------------------------------------------------------------------
                d_fs_dx += (pp[k]*dpg_dx[k] - pg[k]*dpp_dx[k])/den;
                d_fs_dy += (pp[k]*dpg_dy[k] - pg[k]*dpp_dy[k])/den;
                // cout << k << " " << d_invsigmafs_dx << " " << d_invsigmafs_dy << " " << dfs_dx << " " << dfs_dy << endl;
            }

            // Also get variations for IR calculation
            // --------------------------------------
            if (eta_IR!=0.) {
                // Compute variation of integrated resolution with respect to dx, dy
                // We define U_IR = sum_k {PA[k]*Wk*[Et^2/[delta2*Et^2+(Et-Ek)^2]]} / sum_k {PA[k]*Wk}
                // with      Wk = 1. + Wslope*log(Et/Emin)
                // and       deltaEk = Et^2/[delta2*Et^2+(Et-Ek)^2]
                // The derivative wrt xi is then found as
                //           dU/dxi = { num'*den - num*den'/[den^2]}
                // with
                //           num  = sum_k {PA[k]*Wk*[Et^2/[delta2*Et^2+(Et-Ek)^2]]}
                //           den  = sum_k {PA[k]*Wk}
                //           num' = sum_k {dPA[k]/dxi *Wk*deltaEk + 
                //                  2PA[k]*Wk*Et^2*(Et-Ek)/[Et^2*delta2+(Et-Ek)^2]^2) * dEk/dRik * dRik/dxi } 
                //           den' = sum_k {dPA[k]/dxi Wk}
                // --------------------------------------------------------------------------------
                if (IsGamma[k]) {
                    double Em      = Emeas[k][0];
                    double Et      = TrueE[k];
                    double Et2     = Et*Et;
                    double dEk     = Et-Em;
                    double deltaEk = (delta2*Et2+dEk*dEk);
                    double dPAk_dx = 0.;
                    double dPAk_dy = 0.;
                    double Wk      = 1. + Wslope * log(Et/Emin);
                    if (fabs(SumProbgt1[k]-Ntrigger)<1.) {
                        dPAk_dx = ProbTrigger (SumProbgt1[k],1,id,k);
                        dPAk_dy = ProbTrigger (SumProbgt1[k],2,id,k);
                    }
                    sumdPAkdxi += dPAk_dx * Wk;
                    sumdPAkdyi += dPAk_dy * Wk;
                    sumdPAkdxidE2 += dPAk_dx * Wk * Et2/deltaEk; 
                    sumdPAkdyidE2 += dPAk_dy * Wk * Et2/deltaEk; 
                    double dEdR = dEk_dRik(id,k); // here we use the measured energy and R, as we are computing derivatives wrt those
                    double dRdx = -EffectiveDistance (x[id],y[id],x0meas[k][0],y0meas[k][0],thmeas[k][0],phmeas[k][0],1); // nb measured R!
                    double dRdy = -EffectiveDistance (x[id],y[id],x0meas[k][0],y0meas[k][0],thmeas[k][0],phmeas[k][0],2);
                    double factor = 2.*PActive[k] * Wk * Et2 * dEk / pow(deltaEk,2) * dEk_dRik(id,k);
                    sum_dedx += factor*dRdx;
                    sum_dedy += factor*dRdy;
                }
            }
        } // end k loop on batch events
        d_fs_dx *= sigmafs2;
        d_fs_dy *= sigmafs2;

        // Remember to multiply dinvsfs/dx by sigmafs, as the derivative of f(x)^-0.5 is 1/[2*f(x)]* f'(x) 
        // and the sum is f'(x) only. Also, the factor of 2 was taken care of above
        // -----------------------------------------------------------------------------------------------
        d_invsigfs_dx = d_invsigfs_dx * MeasFgErr;
        d_invsigfs_dy = d_invsigfs_dy * MeasFgErr;

        double dIR_dxi = 0.;
        double dIR_dyi = 0.;
        if (eta_IR!=0.) {
            if (U_IR_Den !=0.) {
                dIR_dxi = ((sumdPAkdxidE2 + sum_dedx) * U_IR_Den - U_IR_Num * sumdPAkdxi) / pow(U_IR_Den,2.);
                dIR_dyi = ((sumdPAkdyidE2 + sum_dedy) * U_IR_Den - U_IR_Num * sumdPAkdyi) / pow(U_IR_Den,2.);
            } else {
                warnings5++;
            }
        }

        // Accumulate the derivative of the utility with respect to x,y
        // Note, we use the measured Fg here, but we could opt for a saturated model instead
        // -----------------------------------------------------------------------------------
        // cout << id << " " << d_invsigfs_dx << " " << d_invsigfs_dy << " " << d_fs_dx << " " << d_fs_dy << " invsig = " << inv_sigmafs << endl;
        dU_dxi[id] = eta_GF * (d_invsigfs_dx * MeasFg + d_fs_dx * inv_sigmafs)*ExposureFactor + eta_IR * dIR_dxi;
        dU_dyi[id] = eta_GF * (d_invsigfs_dy * MeasFg + d_fs_dy * inv_sigmafs)*ExposureFactor + eta_IR * dIR_dyi;
#ifdef PLOTS
        //dUdx->Fill(1.*epoch,fabs(dU_dxi[id]));
        //dUdx->Fill(1.*epoch,fabs(dU_dyi[id])); // One histogram for both coordinates
#endif

    } // end id loop on dets

}


// Function that fills parameters of showers
// -----------------------------------------
int ReadShowers () {

#ifdef STANDALONE
    string trainPath  = "/lustre/cmswork/dorigo/swgo/MT/Model/";
#endif
#ifdef INROOT
    string trainPath  = "./SWGO/Model/";
#endif
    ifstream asciifile_g, asciifile_p;

    // Read gamma parameters
    std::stringstream sstr_g;
    sstr_g << "Fit_Photon_10_pars";
    string traininglist_g = trainPath  + sstr_g.str() + ".txt";
    asciifile_g.open(traininglist_g);
    double e;
    for (int jp=0; jp<3; jp++) {
        for (int ip=0; ip<3; ip++) {
            asciifile_g >> e;
            PXeg1_p[jp][ip] = e;
        }
        if (PXeg1_p[jp][0]*PXeg1_p[jp][1]*PXeg1_p[jp][2]==0) {
            cout << "Warning, p" << jp << "eg1 = " << PXeg1_p[jp][0] << " " << PXeg1_p[jp][1] << " " << PXeg1_p[jp][2] << endl;
            return 1;
        }
    }
    for (int jp=0; jp<3; jp++) {
        for (int ip=0; ip<3; ip++) {
            asciifile_g >> e;
            PXeg2_p[jp][ip] = e;
        }
        if (PXeg2_p[jp][0]*PXeg2_p[jp][1]*PXeg2_p[jp][2]==0) {
            cout << "Warning, p" << jp << "eg2 = " << PXeg2_p[jp][0] << " " << PXeg2_p[jp][1] << " " << PXeg2_p[jp][2] << endl;
            return 1;
        }
    }
    for (int jp=0; jp<3; jp++) {
        for (int ip=0; ip<3; ip++) {
            asciifile_g >> e;
            PXeg3_p[jp][ip] = e;
        }
        if (PXeg3_p[jp][0]*PXeg3_p[jp][1]*PXeg3_p[jp][2]==0) {
            cout << "Warning, p" << jp << "eg3 = " << PXeg3_p[jp][0] << " " << PXeg3_p[jp][1] << " " << PXeg3_p[jp][2] << endl;
            return 1;
        }
    }
    for (int jp=0; jp<3; jp++) {
        for (int ip=0; ip<3; ip++) {
            asciifile_g >> e;
            PXeg4_p[jp][ip] = e;
        }
        if (PXeg4_p[jp][0]*PXeg4_p[jp][1]*PXeg4_p[jp][2]==0) {
            cout << "Warning, p" << jp << "eg4 = " << PXeg4_p[jp][0] << " " << PXeg4_p[jp][1] << " " << PXeg4_p[jp][2] << endl;
            return 1;
        }
    }
    for (int jp=0; jp<3; jp++) {
        for (int ip=0; ip<3; ip++) {
            asciifile_g >> e;
            PXmg1_p[jp][ip] = e;
        }
        if (PXmg1_p[jp][0]*PXmg1_p[jp][1]*PXmg1_p[jp][2]==0 && jp!=1) { // par1 is 0 for muons
            cout << "Warning, p" << jp << "mg1 = " << PXmg1_p[jp][0] << " " << PXmg1_p[jp][1] << " " << PXmg1_p[jp][2] << endl;
            return 1;
        }
    }
    for (int jp=0; jp<3; jp++) {
        for (int ip=0; ip<3; ip++) {
            asciifile_g >> e;
            PXmg2_p[jp][ip] = e;
        }
        if (PXmg2_p[jp][0]*PXmg2_p[jp][1]*PXmg2_p[jp][2]==0 && jp!=1) { // par1 is 0 for muons
            cout << "Warning, p" << jp << "mg2 = " << PXmg2_p[jp][0] << " " << PXmg2_p[jp][1] << " " << PXmg2_p[jp][2] << endl;
            return 1;
        }
    }
    for (int jp=0; jp<3; jp++) {
        for (int ip=0; ip<3; ip++) {
            asciifile_g >> e;
            PXmg3_p[jp][ip] = e;
        }
        if (PXmg3_p[jp][0]*PXmg3_p[jp][1]*PXmg3_p[jp][2]==0 && jp!=1) { // par1 is 0 for muons
            cout << "Warning, p" << jp << "mg3 = " << PXmg3_p[jp][0] << " " << PXmg3_p[jp][1] << " " << PXmg3_p[jp][2] << endl;
            return 1;
        }
    }
    for (int jp=0; jp<3; jp++) {
        for (int ip=0; ip<3; ip++) {
            asciifile_g >> e;
            PXmg4_p[jp][ip] = e;
        }
        if (PXmg4_p[jp][0]*PXmg4_p[jp][1]*PXmg4_p[jp][2]==0 && jp!=1) { // par1 is 0 for muons
            cout << "Warning, p" << jp << "mg4 = " << PXmg4_p[jp][0] << " " << PXmg4_p[jp][1] << " " << PXmg4_p[jp][2] << endl;
            return 1;
        }
    }
    asciifile_g.close();

    // Proton data now
    std::stringstream sstr_p;
    sstr_p << "Fit_Proton_2_pars";
    string traininglist_p = trainPath  + sstr_p.str() + ".txt";
    asciifile_p.open(traininglist_p);
    for (int jp=0; jp<3; jp++) {
        for (int ip=0; ip<3; ip++) {
            asciifile_p >> e;
            PXep1_p[jp][ip] = e;
        }
        if (PXep1_p[jp][0]*PXep1_p[jp][1]*PXep1_p[jp][2]==0) {
            cout << "Warning, p" << jp << "ep1 = " << PXep1_p[jp][0] << " " << PXep1_p[jp][1] << " " << PXep1_p[jp][2] << endl;
            warnings1++;
        }
    }
    for (int jp=0; jp<3; jp++) {
        for (int ip=0; ip<3; ip++) {
            asciifile_p >> e;
            PXep2_p[jp][ip] = e;
        }
        if (PXep2_p[jp][0]*PXep2_p[jp][1]*PXep2_p[jp][2]==0) {
            cout << "Warning, p" << jp << "ep2 = " << PXep2_p[jp][0] << " " << PXep2_p[jp][1] << " " << PXep2_p[jp][2] << endl;
            warnings1++;
        }
    }
    for (int jp=0; jp<3; jp++) {
        for (int ip=0; ip<3; ip++) {
            asciifile_p >> e;
            PXep3_p[jp][ip] = e;
        }
        if (PXep3_p[jp][0]*PXep3_p[jp][1]*PXep3_p[jp][2]==0) {
            cout << "Warning, p" << jp << "ep3 = " << PXep3_p[jp][0] << " " << PXep3_p[jp][1] << " " << PXep3_p[jp][2] << endl;
            warnings1++;
            return 1;
        }
    }
    for (int jp=0; jp<3; jp++) {
        for (int ip=0; ip<3; ip++) {
            asciifile_p >> e;
            PXep4_p[jp][ip] = e;
        }
        if (PXep4_p[jp][0]*PXep4_p[jp][1]*PXep4_p[jp][2]==0) {
            cout << "Warning, p" << jp << "ep4 = " << PXep4_p[jp][0] << " " << PXep4_p[jp][1] << " " << PXep4_p[jp][2] << endl;
            warnings1++;
            return 1;
        }
    }
    for (int jp=0; jp<3; jp++) {
        for (int ip=0; ip<3; ip++) {
            asciifile_p >> e;
            PXmp1_p[jp][ip] = e;
        }
        if (PXmp1_p[jp][0]*PXmp1_p[jp][1]*PXmp1_p[jp][2]==0 && jp!=1) { // par1 is 0 for muons
            cout << "Warning, p" << jp << "mp1 = " << PXmp1_p[jp][0] << " " << PXmp1_p[jp][1] << " " << PXmp1_p[jp][2] << endl;
            warnings1++;
            return 1;
        }
    }
    for (int jp=0; jp<3; jp++) {
        for (int ip=0; ip<3; ip++) {
            asciifile_p >> e;
            PXmp2_p[jp][ip] = e;
        }
        if (PXmp2_p[jp][0]*PXmp2_p[jp][1]*PXmp2_p[jp][2]==0 && jp!=1) { // par1 is 0 for muons
            cout << "Warning, p" << jp << "mp2 = " << PXmp2_p[jp][0] << " " << PXmp2_p[jp][1] << " " << PXmp2_p[jp][2] << endl;
            warnings1++;
            return 1;
        }
    }
    for (int jp=0; jp<3; jp++) {
        for (int ip=0; ip<3; ip++) {
            asciifile_p >> e;
            PXmp3_p[jp][ip] = e;
        }
        if (PXmp3_p[jp][0]*PXmp3_p[jp][1]*PXmp3_p[jp][2]==0 && jp!=1) { // par1 is 0 for muons 
            cout << "Warning, p" << jp << "mp3 = " << PXmp3_p[jp][0] << " " << PXmp3_p[jp][1] << " " << PXmp3_p[jp][2] << endl;
            warnings1++;
            return 1;
        }
    }
    for (int jp=0; jp<3; jp++) {
        for (int ip=0; ip<3; ip++) {
            asciifile_p >> e;
            PXmp4_p[jp][ip] = e;
        }
        if (PXmp4_p[jp][0]*PXmp4_p[jp][1]*PXmp4_p[jp][2]==0 && jp!=1) { // par1 is 0 for muons
        cout << "Warning, p" << jp << "mp4 = " << PXmp4_p[jp][0] << " " << PXmp4_p[jp][1] << " " << PXmp4_p[jp][2] << endl;
            warnings1++;
            return 1;
        }
    }
    asciifile_p.close();

    // Initialize the inverse matrix for the cubic interpolation
    // ---------------------------------------------------------
    InitInverse4by4 ();

    // Flux and derivatives lookup table.
    // Obtain parameters in 100x100 grid of energy and theta values
    // ------------------------------------------------------------
    // debugging checks below, ignore next 12 lines
    /*
    Y[0] = 1;
    Y[1] = 4;
    Y[2] = 9;
    Y[3] = 16;
    double val = solvecubic_mg(0,20.,(1.-0.5)*thetamax/4.,0);
    cout << val << endl;
    Y[0] = 0.3253;
    Y[1] = 0.3205;
    Y[2] = 0.3097;
    Y[3] = 0.2909;
    val = solvecubic_mg(0,20.,0.137511,0);
    cout << val << endl;
    */

    for (int ie=0; ie<100; ie++) {
        double energy = 0.1 + 0.1*ie;

        // Convert energy into the function we use in the interpolation
        // ------------------------------------------------------------
        double xe = 0.5+20.*(log(energy)-log_01)/logdif; // energy is in PeV
        double xe2 = xe*xe;
        for (int it=0; it<100; it++) {
            double theta = it*thetamax/99.; // Want to get 0 and 65 degrees
            Y[0] = exp(PXmg1_p[0][0]) + exp(PXmg1_p[0][1]*pow(xe,PXmg1_p[0][2]));
            Y[1] = exp(PXmg2_p[0][0]) + exp(PXmg2_p[0][1]*pow(xe,PXmg2_p[0][2]));
            Y[2] = exp(PXmg3_p[0][0]) + exp(PXmg3_p[0][1]*pow(xe,PXmg3_p[0][2]));
            Y[3] = exp(PXmg4_p[0][0]) + exp(PXmg4_p[0][1]*pow(xe,PXmg4_p[0][2]));
            thisp0_mg[ie][it]     = solvecubic_mg(0,energy,theta,0);
#ifdef PLOTS
            P0mg->SetBinContent(ie+1,it+1,thisp0_mg[ie][it]);
#endif
            dthisp0de_mg[ie][it]  = solvecubic_mg(0,energy,theta,2);
            dthisp0dth_mg[ie][it] = solvecubic_mg(0,energy,theta,3);
            Y[0] = PXmg1_p[2][0] + PXmg1_p[2][1]*xe + PXmg1_p[2][2]*xe2;
            Y[1] = PXmg2_p[2][0] + PXmg2_p[2][1]*xe + PXmg2_p[2][2]*xe2;
            Y[2] = PXmg3_p[2][0] + PXmg3_p[2][1]*xe + PXmg3_p[2][2]*xe2;
            Y[3] = PXmg4_p[2][0] + PXmg4_p[2][1]*xe + PXmg4_p[2][2]*xe2;
            thisp2_mg[ie][it]     = solvecubic_mg(2,energy,theta,0);
#ifdef PLOTS
            P2mg->SetBinContent(ie+1,it+1,thisp2_mg[ie][it]);
#endif
            dthisp2de_mg[ie][it]  = solvecubic_mg(2,energy,theta,2);
            dthisp2dth_mg[ie][it] = solvecubic_mg(2,energy,theta,3);

            Y[0] = PXeg1_p[0][0] * exp(PXeg1_p[0][1] * pow(xe,PXeg1_p[0][2]));
            Y[1] = PXeg2_p[0][0] * exp(PXeg2_p[0][1] * pow(xe,PXeg2_p[0][2]));
            Y[2] = PXeg3_p[0][0] * exp(PXeg3_p[0][1] * pow(xe,PXeg3_p[0][2]));
            Y[3] = PXeg4_p[0][0] * exp(PXeg4_p[0][1] * pow(xe,PXeg4_p[0][2]));
            thisp0_eg[ie][it]     = solvecubic_eg(0,energy,theta,0);
#ifdef PLOTS
            P0eg->SetBinContent(ie+1,it+1,thisp0_eg[ie][it]);
#endif
            dthisp0de_eg[ie][it]  = solvecubic_eg(0,energy,theta,2);
            dthisp0dth_eg[ie][it] = solvecubic_eg(0,energy,theta,3);
            Y[0] = PXeg1_p[1][0] + PXeg1_p[1][1]*xe + PXeg1_p[1][2]*xe2;
            Y[1] = PXeg2_p[1][0] + PXeg2_p[1][1]*xe + PXeg2_p[1][2]*xe2;
            Y[2] = PXeg3_p[1][0] + PXeg3_p[1][1]*xe + PXeg3_p[1][2]*xe2;
            Y[3] = PXeg4_p[1][0] + PXeg4_p[1][1]*xe + PXeg4_p[1][2]*xe2;
            thisp1_eg[ie][it]     = solvecubic_eg(1,energy,theta,0);
#ifdef PLOTS
            P1eg->SetBinContent(ie+1,it+1,thisp1_eg[ie][it]);
#endif
            dthisp1de_eg[ie][it]  = solvecubic_eg(1,energy,theta,2);
            dthisp1dth_eg[ie][it] = solvecubic_eg(1,energy,theta,3);
            Y[0] = PXeg1_p[2][0] + PXeg1_p[2][1]*xe + PXeg1_p[2][2]*xe2;
            Y[1] = PXeg2_p[2][0] + PXeg2_p[2][1]*xe + PXeg2_p[2][2]*xe2;
            Y[2] = PXeg3_p[2][0] + PXeg3_p[2][1]*xe + PXeg3_p[2][2]*xe2;
            Y[3] = PXeg4_p[2][0] + PXeg4_p[2][1]*xe + PXeg4_p[2][2]*xe2;
            thisp2_eg[ie][it]     = solvecubic_eg(2,energy,theta,0);
#ifdef PLOTS
            P2eg->SetBinContent(ie+1,it+1,thisp2_eg[ie][it]);
#endif
            dthisp2de_eg[ie][it]  = solvecubic_eg(2,energy,theta,2);
            dthisp2dth_eg[ie][it] = solvecubic_eg(2,energy,theta,3);

            Y[0] = exp(PXmp1_p[0][0]) + exp(PXmp1_p[0][1]*pow(xe,PXmp1_p[0][2]));
            Y[1] = exp(PXmp2_p[0][0]) + exp(PXmp2_p[0][1]*pow(xe,PXmp2_p[0][2]));
            Y[2] = exp(PXmp3_p[0][0]) + exp(PXmp3_p[0][1]*pow(xe,PXmp3_p[0][2]));
            Y[3] = exp(PXmp4_p[0][0]) + exp(PXmp4_p[0][1]*pow(xe,PXmp4_p[0][2]));
            thisp0_mp[ie][it]     = solvecubic_mp(0,energy,theta,0);
#ifdef PLOTS
            P0mp->SetBinContent(ie+1,it+1,thisp0_mp[ie][it]);
#endif
            dthisp0de_mp[ie][it]  = solvecubic_mp(0,energy,theta,2);
            dthisp0dth_mp[ie][it] = solvecubic_mp(0,energy,theta,3);
            Y[0] = PXmp1_p[2][0] + PXmp1_p[2][1]*xe + PXmp1_p[2][2]*xe2;
            Y[1] = PXmp2_p[2][0] + PXmp2_p[2][1]*xe + PXmp2_p[2][2]*xe2;
            Y[2] = PXmp3_p[2][0] + PXmp3_p[2][1]*xe + PXmp3_p[2][2]*xe2;
            Y[3] = PXmp4_p[2][0] + PXmp4_p[2][1]*xe + PXmp4_p[2][2]*xe2;
            thisp2_mp[ie][it]     = solvecubic_mp(2,energy,theta,0);
#ifdef PLOTS
            P2mp->SetBinContent(ie+1,it+1,thisp2_mp[ie][it]);
#endif
            dthisp2de_mp[ie][it]  = solvecubic_mp(2,energy,theta,2);
            dthisp2dth_mp[ie][it] = solvecubic_mp(2,energy,theta,3);

            Y[0] = exp(PXep1_p[0][0]) + exp(PXep1_p[0][1]*pow(xe,PXep1_p[0][2]));
            Y[1] = exp(PXep2_p[0][0]) + exp(PXep2_p[0][1]*pow(xe,PXep2_p[0][2]));
            Y[2] = exp(PXep3_p[0][0]) + exp(PXep3_p[0][1]*pow(xe,PXep3_p[0][2]));
            Y[3] = exp(PXep4_p[0][0]) + exp(PXep4_p[0][1]*pow(xe,PXep4_p[0][2]));
            thisp0_ep[ie][it]     = solvecubic_ep(0,energy,theta,0);
#ifdef PLOTS
            P0ep->SetBinContent(ie+1,it+1,thisp0_ep[ie][it]);
#endif
            dthisp0de_ep[ie][it]  = solvecubic_ep(0,energy,theta,2);
            dthisp0dth_ep[ie][it] = solvecubic_ep(0,energy,theta,3);
            Y[0] = PXep1_p[1][0] + PXep1_p[1][1]*xe + PXep1_p[1][2]*xe2;
            Y[1] = PXep2_p[1][0] + PXep2_p[1][1]*xe + PXep2_p[1][2]*xe2;
            Y[2] = PXep3_p[1][0] + PXep3_p[1][1]*xe + PXep3_p[1][2]*xe2;
            Y[3] = PXep4_p[1][0] + PXep4_p[1][1]*xe + PXep4_p[1][2]*xe2;
            thisp1_ep[ie][it]     = solvecubic_ep(1,energy,theta,0);
#ifdef PLOTS
            P1ep->SetBinContent(ie+1,it+1,thisp1_ep[ie][it]);
#endif
            dthisp1de_ep[ie][it]  = solvecubic_ep(1,energy,theta,2);
            dthisp1dth_ep[ie][it] = solvecubic_ep(1,energy,theta,3);
            Y[0] = PXep1_p[2][0] + PXep1_p[2][1]*xe + PXep1_p[2][2]*xe2;
            Y[1] = PXep2_p[2][0] + PXep2_p[2][1]*xe + PXep2_p[2][2]*xe2;
            Y[2] = PXep3_p[2][0] + PXep3_p[2][1]*xe + PXep3_p[2][2]*xe2;
            Y[3] = PXep4_p[2][0] + PXep4_p[2][1]*xe + PXep4_p[2][2]*xe2;
            thisp2_ep[ie][it]     = solvecubic_ep(2,energy,theta,0);
#ifdef PLOTS
            P2ep->SetBinContent(ie+1,it+1,thisp2_ep[ie][it]);
#endif
            dthisp2de_ep[ie][it]  = solvecubic_ep(2,energy,theta,2);
            dthisp2dth_ep[ie][it] = solvecubic_ep(2,energy,theta,3);
        }
    }
    return 0;
}

// Determine the measured gamma fraction and a lower bound on its variance
// -----------------------------------------------------------------------
double MeasuredGammaFraction () {
    double MeasFg = 0.5;
    double dlnL_dfg_orig, dlnL_dfg, dlnL_dfg_new;
    double num; 
    double den;
    int Nloops = 0;
    do {
        dlnL_dfg     = 0.;
        inv_sigmafs2 = 0.;
        for (int k=Nevents; k<Nevents+Nbatch; k++) {
            if (Active[k]) {
                num = pg[k]-pp[k];
                den = (MeasFg*pg[k]+(1.-MeasFg)*pp[k]);
                if (den>0.) {
                    dlnL_dfg     += num/den;
                    inv_sigmafs2 += num*num/(den*den);
                }
            }
        }
        if (Nloops==0) dlnL_dfg_orig = dlnL_dfg;

        // Since we get dlnL/dfg = x !=0, and we want it to be =0, we need to modify MeasFg.
        // The variation dMeasFg is found by noticing that we want dlnL/dMeasFg to change by -x,
        // and such a change for a change of dMeasFg equals the second derivative dlnL/dMeasFg,
        // i.e. minus the inverse of sigmafg^2. So -x/dMeasFg = -1/sigmaMeasFg^2 from which we
        // get the expression below.
        // -----------------------------------------------------------------------------
        MeasFg += dlnL_dfg/inv_sigmafs2;

        // Check if we get close to zero with dlnL/dMeasFg now
        // -----------------------------------------------
        dlnL_dfg_new = 0.;
        inv_sigmafs2 = 0.;
        for (int k=Nevents; k<Nevents+Nbatch; k++) {
            if (Active[k]) {
                num = pg[k]-pp[k];
                den = (MeasFg*pg[k]+(1.-MeasFg)*pp[k]);
                if (den>0.) {
                    dlnL_dfg_new += num/den;
                    inv_sigmafs2 += num*num/(den*den); // This is static and will be used later
                }
            }
        }
        MeasFg += dlnL_dfg_new/inv_sigmafs2;
        Nloops++;
    } while (fabs(dlnL_dfg_new)>0.001);
    // cout << "     After " << Nloops << " loops, calc of measured Fg gives " << MeasFg 
    //      << " with dlnL/dFg = " << dlnL_dfg_new << " (for 0.5 was " << dlnL_dfg_orig << ")" << endl;
    return MeasFg;
}

// Dump warnings if terminating
// ----------------------------
void TerminateAbnormally () {
    cout    << "     There were serious warnings, terminating. " << endl;
    cout    << "     Warnings: " << endl;
    cout    << "     1 - " << warnings1 << endl;
    cout    << "     2 - " << warnings2 << endl;
    cout    << "     3 - " << warnings3 << endl;
    cout    << "     4 - " << warnings4 << endl;
    cout    << "     5 - " << warnings5 << endl;
    cout    << "     6 - " << warnings6 << endl;
    cout    << "--------------------------------------------------------------" << endl;

    // Close dump file
    // ---------------
    outfile << endl;
    outfile << "The program terminated due to warnings. " << endl;
    outfile << "     Warnings: " << endl;
    outfile << "     1 - " << warnings1 << endl;
    outfile << "     2 - " << warnings2 << endl;
    outfile << "     3 - " << warnings3 << endl;
    outfile << "     4 - " << warnings4 << endl;
    outfile << "     5 - " << warnings5 << endl;
    outfile << "     6 - " << warnings6 << endl;
    outfile << "--------------------------------------------------------------" << endl;
    outfile.close();
    return;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//                                                              Main routine
//
// -------------------------------------------------------------------------------------------------------------------------------------
#ifdef STANDALONE
int main (int argc, char * argv[]) {

    // Default values of pass parameters
    // ---------------------------------
    Nevents            = 2100;
    Nbatch             = 2100;
    Nunits             = 91;
    Nepochs            = 500;
    DetectorSpacing    = 50;
    SpacingStep        = 50;
    Rslack             = 1000;
    shape              = 3;
    commonMode         = 0;
    StartLR            = 1.0;
    Ngrid              = 100;
    NEgrid             = 10;
    maxNsteps          = 500;
    LRX                = 5.;
    LRA                = 0.02;
    LRE                = 0.05;
    Nthreads           = 30;
    eta_GF             = 100.;
    eta_IR             = 1000.;
    startEpoch         = 0.;
    Ntrigger           = 5;
    OrthoShowers       = false;
    SlantedShowers     = false;
    usetrueXY          = false;
    usetrueAngs        = false;
    usetrueE           = false;
    scanU              = false;
    readGeom           = false;

    // Command line arguments
    // ----------------------
    for (int i=0; i<argc; i++) {
        if (!strcmp(argv[i],"-h")) {
            cout << "List of arguments:" << endl;
            cout << "-nev number of events for PDF generation (expects int)" << endl;
            cout << "-nba number of batch events (expects int)" << endl;
            cout << "-nde number of detector units (expects int)" << endl;
            cout << "-nep number of training epochs (expects int)" << endl;
            cout << "-spa spacing between detectors in meters (expects float)" << endl;
            cout << "-sst step between elements in meters (expects float)" << endl;
            cout << "-rsl extra radial space in generated shower area (expects float)" << endl;
            cout << "-sha shape of initial layout (0-9, 101-114) (expects int)" << endl;
            cout << "-com common mode (0 independent movement, 1,2 common modes)" << endl;
            cout << "-slr start learning rate (expects float in 0.1:10.)" << endl;
            cout << "-ngr number of grid search points in xy in shower likelihood (expects int)" << endl;
            cout << "-nei number of energy initializations (expects int)" << endl;
            cout << "-nst max number of steps in shower likelihood (expects int)" << endl;
            cout << "-lre learning rate for energy (expects float)" << endl;
            cout << "-lrx learning rate for position (expects float)" << endl;
            cout << "-lra learning rate for angles (expects float)" << endl;
            cout << "-nth number of threads (expects int)" << endl;
            cout << "-etf utility weight of flux uncertainty (def. 1.0)" << endl;
            cout << "-ete utility weight of integrated energy resolution (use neg. value, def. -0.3)" << endl;
            cout << "-ntr minimum number of detectors hit by triggering showers" << endl;
            cout << "-utx use true (and do not fit) XY of showers" << endl;
            cout << "-uta use true (and do not fit) angle of showers" << endl;
            cout << "-ute use true (and do not fit) energy of showers" << endl;
            cout << "-ort use orthogonal showers (theta=0, phi undef)" << endl;
            cout << "-sla use showers at theta=pi/4" << endl;
            cout << "-sca scan utility function around one point" << endl;
            cout << "-rea read geometry from previous run" << endl;
            cout << "-ste starting epoch (use to continue previous runs, expects int)" << endl;
            return 0;
        }
        else if (!strcmp(argv[i],"-nev")) {Nevents         = atoi(argv[++i]);}
        else if (!strcmp(argv[i],"-nba")) {Nbatch          = atoi(argv[++i]);}
        else if (!strcmp(argv[i],"-nde")) {Nunits          = atoi(argv[++i]);}
        else if (!strcmp(argv[i],"-nep")) {Nepochs         = atoi(argv[++i]);}
        else if (!strcmp(argv[i],"-spa")) {DetectorSpacing = atof(argv[++i]);}
        else if (!strcmp(argv[i],"-sst")) {SpacingStep     = atof(argv[++i]);}
        else if (!strcmp(argv[i],"-rsl")) {Rslack          = atof(argv[++i]);}
        else if (!strcmp(argv[i],"-sha")) {shape           = atoi(argv[++i]);}
        else if (!strcmp(argv[i],"-com")) {commonMode      = atoi(argv[++i]);}
        else if (!strcmp(argv[i],"-slr")) {StartLR         = atof(argv[++i]);}
        else if (!strcmp(argv[i],"-ngr")) {Ngrid           = atoi(argv[++i]);}
        else if (!strcmp(argv[i],"-nei")) {NEgrid          = atoi(argv[++i]);}
        else if (!strcmp(argv[i],"-nst")) {maxNsteps       = atoi(argv[++i]);}
        else if (!strcmp(argv[i],"-lre")) {LRE             = atof(argv[++i]);}
        else if (!strcmp(argv[i],"-lrx")) {LRX             = atof(argv[++i]);}
        else if (!strcmp(argv[i],"-lra")) {LRA             = atof(argv[++i]);}
        else if (!strcmp(argv[i],"-nth")) {Nthreads        = atoi(argv[++i]);}
        else if (!strcmp(argv[i],"-etf")) {eta_GF          = atof(argv[++i]);}
        else if (!strcmp(argv[i],"-ete")) {eta_IR          = atof(argv[++i]);}
        else if (!strcmp(argv[i],"-ste")) {startEpoch      = atoi(argv[++i]);}
        else if (!strcmp(argv[i],"-ete")) {Ntrigger        = atoi(argv[++i]);}
        else if (!strcmp(argv[i],"-utx")) {usetrueXY       = true;}
        else if (!strcmp(argv[i],"-uta")) {usetrueAngs     = true;}
        else if (!strcmp(argv[i],"-ute")) {usetrueE        = true;}    
        else if (!strcmp(argv[i],"-ort")) {OrthoShowers    = true;}    
        else if (!strcmp(argv[i],"-sla")) {SlantedShowers  = true;}    
        else if (!strcmp(argv[i],"-sca")) {scanU           = true;}    
        else if (!strcmp(argv[i],"-rea")) {readGeom        = true;}    
    }  
#endif // STANDALONE

    // Pass parameters:
    // ----------------
    // Nevents          = number of generated showers for templates generation (choose an even number to have same # of p and g showers)
    // Nbatch           = number of showers per batch in gradient descent. Note, this number should usually be equal to Nevents and EVEN 
    // Nunits           = number of detector elements. For radial distr, use 1/7/19/37/61/91/127/169/217/271/331/397/469/547/631/721...
    // Nepochs          = number of SGD loops
    // DetectorSpacing  = initial spacing of tanks
    // SpacingStep      = increase in spacing 
    // Rslack           = space of showers away from detector units
    // shape            = geometry of the initial layout (0=hexagonal, 1=taxi, 2=spiral)
    // commonMode       = whether xy of units is varied independently (0), or radius (1) or offset (2)
    // StartLR          = starting learning rate for grad descent
    // Ngrid            = number of grid points on the plane for initial assay of likelihood value
    // NEgrid           = number of energy points in [Emin,Emax] for initial assay of likelihood value
    // LRE              = learning rate multiplier of dlogL/dE in likelihood maximization
    // LRX              = learning rate multiplier of dlogL/dX in likelihood maximization
    // LRA              = learning rate multiplier of dlogL/dth, ph in likelihood maximization
    // Nthreads         = number of threads for multi-CPU use
    // eta_GF           = utility multiplier for gamma flux uncertainty (def. 100)
    // eta_IR           = utility multiplier for integrated energy resolution (def. 1000)
    // startEpoch       = starting epoch (for continuing runs with readGeom true, to not overwrite plots)
    // Ntrigger         = minimum number of detectors hit by accepted shower
    // usetrueXY        = if true xy of showers is not fit, true values used
    // usetrueAngs      = if true angles are not fit, true value used
    // usetrueE         = if true energy of showers is not fit, true value used
    // OrthoShowers     = if true, showers are generated with zero polar angle 
    // SlantedShowers   = if true, showers are generated at pi/4
    // scanU            = if true, a scan of the utility is performed around one point
    // readGeom         = if true, the geometry is read in from file with corresponding parameters

#ifdef INROOT
int swgolo (int Nev=1000, int Nba=1000, int Nu=37, int Nep=200, 
             double Spacing=50., double SpSt = 50., double Rsl=300., int sh=3, 
             int cm=0, double SLR = 1.0, bool OS = true, bool uTA = true, bool uTE = false, int Ntr = 5, 
             int Ngr = 25, int NEi = 10, double LR_E=0.01, double LR_X = 0.5, double LR_A = 0.01) {
    int Nthr = 1; // no multithreading if running in root
#endif

    // UNITS
    // -----
    // position: meters
    // angle:    radians
    // time:     nanoseconds
    // energy:   PeV

#ifdef INROOT
    // Get static values from pass parameters
    // --------------------------------------
    Nevents         = Nev;
    Nepochs         = Nep;
    Nbatch          = Nba;
    Nunits          = Nu;
    DetectorSpacing = Spacing;
    SpacingStep     = SpSt;
    Rslack          = Rsl;
    shape           = sh;
    commonMode      = cm;
    OrthoShowers    = OS;
    usetrueAngs     = uTA;
    usetrueE        = uTE;
    StartLR         = SLR;
    Ngrid           = Ngr;
    NEgrid          = NEi;
    LRE             = LR_E;
    LRX             = LR_X;
    LRA             = LR_A;
    Ntrigger        = Ntr;
    Nthreads        = Nthr;
#endif

    // Override number of detectors for SWGO shapes
    // --------------------------------------------
    if (shape>100 && shape<114) {
        Nunits = N_predef[shape-101];
    }

    // Safety checks
    // -------------
    if (SameShowers) {
        if (Nevents%2!=0) Nevents++; // we need even Nevents in that case
        Nbatch = Nevents;
        cout << "     SameShowers is on, fixed Nevents = Nbatch = " << Nevents << endl;
    }
    if (StartLR<MinLearningRate || StartLR>MaxLearningRate) {
        cout << "     Learning rate outside range [" << MinLearningRate << "," << MaxLearningRate << "]. Set to 0.1" << endl;
        StartLR = 0.1;
    }
    if (Nunits<minUnits) {
        Nunits = 10;
        cout << "     Too few units. Set to 10." << endl;
    }
    if (Nunits>maxUnits) {
        cout << "     Too many units. Stopping." << endl;
        return 0;
    }
    if (Nevents+Nbatch>maxEvents) {
        cout << "     Too many events. Stopping." << endl;
        return 0;
    }
    if (Nepochs>maxEpochs) {
        cout << "     Too many epochs. Stopping." << endl;
        return 0;
    }
    if (DetectorSpacing<=0.) {
        cout << "     DetectorSpacing must be >0. Stopping." << endl;
        return 0;
    }
    if (SameShowers && Nevents<Nbatch) {
        cout << "     Must have Nevents>=Nbatch if SameShowers is true. Stopping." << endl;
        return 0;
    }
    if (SameShowers && !fixShowerPos && Nthreads>1) {
        cout << "     Sorry, cannot run with SameShowers in multithreading mode. Stopping." << endl;
        return 0;
    }
    if (commonMode>0 && Nthreads>1) {
        cout << "     Sorry, cannot run with commonMode>0 in multithreading mode. Stopping." << endl;
        return 0;
    }

    if (scanU) {
        if (!SameShowers) {
            cout << "     With scanU you must set SameShowers true. Stopping." << endl;
            return 0;
        }
        if (!fixShowerPos) {
            cout << "     With scanU you must set fixShowerPos true. I will do that for you." << endl;
            cout << endl;
            fixShowerPos = true;
        }
    }
    if (OrthoShowers && SlantedShowers) {
        cout << "     Sorry, cannot have both OrthoShowers and SlantedShowers on. Turning OrthoShowers off." << endl;
        OrthoShowers = false;
    }
    if (usetrueE && eta_IR!=0.) {
        cout << "     Using true energy, so I will set eta_IR = 0." << endl;
        eta_IR = 0.;
    }

    // As long as Ntrigger is not large, we can initialize the factorials here
    // -----------------------------------------------------------------------
    if (Ntrigger>maxNtrigger) {
        cout << "     Ntrigger is too large. Terminating. " << endl;
        return 0;
    } else {
        for (int i=0; i<maxNtrigger; i++) {
            F[i] = Factorial(i);
        }
    }

    // Other checks
    // ------------
    if (shape==3 && SpacingStep==0.) {
        cout << "     Sorry, for circular shapes you need to define a radius increment larger than zero!" << endl;
        return 0;
    }

    // Set up output file
    // ------------------
#ifdef STANDALONE
    string outPath  = "/lustre/cmswork/dorigo/swgo/MT/Outputs/";
#endif
#ifdef INROOT
    string outPath = "./SWGO/Outputs/";
#endif
 

    // Determine first available file number to write
    // ----------------------------------------------
    indfile = -1;
    ifstream tmpfile;
    char num[100];
    sprintf (num, "Nb=%d_Nu=%d_Ne=%d-%d_Sh=%d", Nbatch, Nunits, startEpoch, startEpoch+Nepochs, shape);
     do {
        if (indfile>-1) tmpfile.close();
        indfile++;
        std::stringstream tmpstring;
        tmpstring << "RunDetails_" << num << "_" << indfile;
        string tmpfilename = outPath + tmpstring.str() + ".txt";
        tmpfile.open(tmpfilename);
    } while (tmpfile.is_open());

    // Create the outfile for dump of event information
    // ------------------------------------------------
    std::stringstream sstr5; // This one includes the index
    sstr5 << "RunDetails_" << num << "_" << indfile;
    string dump = outPath + sstr5.str() + ".txt";
    outfile.open(dump,ios::app);

    outfile << endl;
    outfile << "     *****************************************************************" << endl;
    outfile << endl;
    outfile << "                          S   W   G   O   L   O                       " << endl;
    outfile << endl; 
    outfile << "         Southern Wide-field Gamma Observatory Layout Optimization    " << endl;
    outfile << endl;
    outfile << "         Proof-of-principle study                                     " << endl;
    outfile << "         of SWGO detector optimization with end-to-end model          " << endl;
    outfile << endl;
    outfile << "                                    T. Dorigo, Oct 2022 - Aug 2023    " << endl;
    outfile << endl;
    outfile << "     *****************************************************************" << endl;
    outfile << endl;
    outfile << "     Running with the following parameters: " << endl;
    outfile << "     initBitmap      = " << initBitmap << endl;
    outfile << "     usetrueXY       = " << usetrueXY << endl;
    outfile << "     usetrueAngs     = " << usetrueAngs << endl;
    outfile << "     usetrueE        = " << usetrueE << endl;
    outfile << "     fixShowerPos    = " << fixShowerPos << endl;
    outfile << "     OrthoShowers    = " << OrthoShowers << endl;
    outfile << "     SlantedShowers  = " << SlantedShowers << endl;
    outfile << "     hexaShowers     = " << hexaShowers << endl;
    outfile << "     SameShowers     = " << SameShowers << endl;
    outfile << "     scanU           = " << scanU << endl;
    outfile << "     idstar          = " << idstar << endl;
    outfile << "     readGeom        = " << readGeom << endl;
    outfile << "     writeGeom       = " << writeGeom << endl;
    outfile << "     Nunits          = " << Nunits << endl;
    outfile << "     Nbatch          = " << Nbatch << endl;
    outfile << "     Nevents         = " << Nevents << endl;
    outfile << "     Nepochs         = " << Nepochs << endl;
    outfile << "     Nthreads        = " << Nthreads << endl;
    outfile << "     maxNsteps       = " << maxNsteps << endl;
    outfile << "     Rmin            = " << Rmin << endl;
    outfile << "     Rslack          = " << Rslack << endl;
    outfile << "     commonMode      = " << commonMode << endl;
    outfile << "     TankArea/pi     = " << TankArea/pi << endl;
    outfile << "     eta_GF          = " << eta_GF << endl;
    outfile << "     eta_IR          = " << eta_IR << endl;
    outfile << "     Wslope          = " << Wslope << endl;
    outfile << "     Eslope          = " << Eslope << endl;
    outfile << "     DetectorSpacing = " << DetectorSpacing << endl;
    outfile << "     SpacingStep     = " << SpacingStep << endl;
    outfile << "     Ngrid           = " << Ngrid << endl;
    outfile << "     NEgrid          = " << NEgrid << endl;
    outfile << "     shape           = " << shape << endl;
    outfile << "     sigma_time      = " << sigma_time << endl;
    outfile << "     sigma_texp      = " << sigma_texp << endl;
    outfile << "     LRE             = " << LRE << endl;
    outfile << "     LRX             = " << LRX << endl;
    outfile << "     LRA             = " << LRA << endl;
    outfile << "     logLApprox      = " << logLApprox << endl;
    outfile << "     StartLR         = " << StartLR << endl;
    outfile << "     Ntrigger        = " << Ntrigger << endl;
    outfile << endl;
    outfile << "     *****************************************************************" << endl;

    cout    << endl;
    cout    << endl;
    cout    << "     *****************************************************************" << endl;
    cout    << endl;
    cout    << "                          S   W   G   O   L   O                       " << endl;
    cout    << endl; 
    cout    << "         Southern Wide-field Gamma Observatory Layout Optimization    " << endl;
    cout    << endl;
    cout    << "         Proof-of-principle study                                     " << endl;
    cout    << "         of SWGO detector optimization with end-to-end model          " << endl;
    cout    << endl;
    cout    << "                                    T. Dorigo, Oct 2022 - Aug 2023    " << endl;
    cout    << endl;
    cout    << "     *****************************************************************" << endl;
    cout    << endl;

    gStyle->SetOptStat(0);
    gStyle->SetOptFit(0);

    // Get a sound RN generator
    // ------------------------
    delete gRandom;
    myRNG = new TRandom3();

    // Suppress root warnings
    // ----------------------
    gROOT->ProcessLine ("gErrorIgnoreLevel = 6001;");
    gROOT->ProcessLine ("gPrintViaErrorHandler = kTRUE;");
     
    // Define the current geometry 
    // ---------------------------
    if (readGeom) {
        ReadLayout ();
    } else {
        DefineLayout (DetectorSpacing,SpacingStep); // -> also defines initial spanR
    }
    // Define initial positions (used to fill Rdistr0 histogram while varying spanR)
    // -----------------------------------------------------------------------------
    for (int id=0; id<Nunits; id++) {
        xinit[id] = x[id];
        yinit[id] = y[id];
    }

    // Fill vector of Gaussian shifts for histogramming of LLR PDFs
    // ------------------------------------------------------------
    for (int i=0; i<10000; i++) {
        shift[i] = myRNG->Gaus(0.,1.);
    }
    NumAvgSteps = 0;
    DenAvgSteps = 0;

    // Define number of R bins depending on initial value of spanR (which is now defined) and Rslack
    // ---------------------------------------------------------------------------------------------
    NRbins = 100*(spanR+Rslack)/spanR;
    if (NRbins>maxRbins) NRbins=maxRbins;

    TrueGammaFraction = 0.5; // But some events may end up becoming inactive, see below.
    
    // Read in parametrizations of particle fluxes and lookup table
    // ------------------------------------------------------------
    int code = ReadShowers ();
    if (code!=0) {
        cout    << "     Unsuccessful retrieval of shower parameters, terminating." << endl;
        outfile << "     Unsuccessful retrieval of shower parameters, terminating." << endl;
        outfile.close();
        return 0;
    }
#ifdef PLOTS
    // Define canvases for check of model
    // ----------------------------------
#endif
    if (checkmodel) {
        TCanvas * TMPe0  = new TCanvas ("TMPe0","", 800,800); 
        TCanvas * TMPe1  = new TCanvas ("TMPe1","", 800,800); 
        TCanvas * TMPe2  = new TCanvas ("TMPe2","", 800,800); 
        TCanvas * TMPe3  = new TCanvas ("TMPe3","", 800,800); 
        TCanvas * TMPm0  = new TCanvas ("TMPm0","", 800,800); 
        TCanvas * TMPm1  = new TCanvas ("TMPm1","", 800,800); 
        TCanvas * TMPm2  = new TCanvas ("TMPm2","", 800,800); 
        TCanvas * TMPm3  = new TCanvas ("TMPm3","", 800,800); 
        TCanvas * mgflux = new TCanvas ("mgflux","",500,500);
        TCanvas * mpflux = new TCanvas ("mpflux","",500,500);
        TCanvas * egflux = new TCanvas ("egflux","",500,500);
        TCanvas * epflux = new TCanvas ("epflux","",500,500);
        TH1D * Rg_e[80];
        TH1D * Rg_m[80];
        TH1D * Rp_e[80];
        TH1D * Rp_m[80];
        char hname[50];
        for (int i=0; i<80; i++) {
            sprintf (hname, "Rg_m%d",i);
            Rg_m[i] = new TH1D (hname,hname,1500, 0., 1500.); 
            sprintf (hname, "Rg_e%d",i);
            Rg_e[i] = new TH1D (hname,hname,1500, 0., 1500.); 
            sprintf (hname, "Rp_m%d",i);
            Rp_m[i] = new TH1D (hname,hname,1500, 0., 1500.); 
            sprintf (hname, "Rp_e%d",i);
            Rp_e[i] = new TH1D (hname,hname,1500, 0., 1500.); 
            Rp_e[i]->SetLineColor(kRed);
            Rp_m[i]->SetLineColor(kRed);
        }
        // Fill histograms of radial densities
        // -----------------------------------
        for (int ie=0; ie<20; ie++) {
            double e = exp((log_10-log_01)*(ie+0.5)/20. + log_01); 
            for (int it=0; it<4; it++) {
                // 0.5 = 0., 4.5 = thetamax. We get values at 1,2,3,4
                double t = thetamax/4.*(it+0.5);
                for (int i=0; i<1500; i++) {
                    double r = 0.5+i;
                    Rg_m[ie*4+it]->SetBinContent (i+1,MFromG(e,t,r,0)/TankArea);
                    Rg_e[ie*4+it]->SetBinContent (i+1,EFromG(e,t,r,0)/TankArea);
                    Rp_m[ie*4+it]->SetBinContent (i+1,MFromP(e,t,r,0)/TankArea);
                    Rp_e[ie*4+it]->SetBinContent (i+1,EFromP(e,t,r,0)/TankArea);
                    if (i%300==0) cout << "mg energy = " << e << " theta = " << t << " R = " << i+0.5 << " " << MFromG(e,t,r,0)/TankArea << endl;
                    if (i%300==0) cout << "eg energy = " << e << " theta = " << t << " R = " << i+0.5 << " " << EFromG(e,t,r,0)/TankArea << endl;
                    if (i%300==0) cout << "mp energy = " << e << " theta = " << t << " R = " << i+0.5 << " " << MFromP(e,t,r,0)/TankArea << endl;
                    if (i%300==0) cout << "ep energy = " << e << " theta = " << t << " R = " << i+0.5 << " " << EFromP(e,t,r,0)/TankArea << endl;
                }
            }   
        }

#ifdef PLOTS
        // Plot them
        // ---------
        TMPe0->Divide(4,5);
        TMPe1->Divide(4,5);
        TMPe2->Divide(4,5);
        TMPe3->Divide(4,5);
        TMPm0->Divide(4,5);
        TMPm1->Divide(4,5);
        TMPm2->Divide(4,5);
        TMPm3->Divide(4,5);
        for (int i=1; i<=20; i++) {
            TMPe0->cd(i);
            TMPe0->GetPad(i)->SetLogy();
            Rg_e[i*4-4]->Draw();
            Rp_e[i*4-4]->Draw("SAME");
        }
        for (int i=1; i<=20; i++) {
            TMPe1->cd(i);
            TMPe1->GetPad(i)->SetLogy();
            Rg_e[i*4-3]->Draw();
            Rp_e[i*4-3]->Draw("SAME");
        }
        for (int i=1; i<=20; i++) {
            TMPe2->cd(i);
            TMPe2->GetPad(i)->SetLogy();
            Rg_e[i*4-2]->Draw();
            Rp_e[i*4-2]->Draw("SAME");
        }
        for (int i=1; i<=20; i++) {
            TMPe3->cd(i);
            TMPe3->GetPad(i)->SetLogy();
            Rg_e[i*4-1]->Draw();
            Rp_e[i*4-1]->Draw("SAME");
        }
        for (int i=1; i<=20; i++) {
            TMPm0->cd(i);
            TMPm0->GetPad(i)->SetLogy();
            Rg_m[i*4-4]->Draw();
            Rp_m[i*4-4]->Draw("SAME");
        }
        for (int i=1; i<=20; i++) {
            TMPm1->cd(i);
            TMPm1->GetPad(i)->SetLogy();
            Rg_m[i*4-3]->Draw();
            Rp_m[i*4-3]->Draw("SAME");
        }
        for (int i=1; i<=20; i++) {
            TMPm2->cd(i);
            TMPm2->GetPad(i)->SetLogy();
            Rg_m[i*4-2]->Draw();
            Rp_m[i*4-2]->Draw("SAME");
        }
        for (int i=1; i<=20; i++) {
            TMPm3->cd(i);
            TMPm3->GetPad(i)->SetLogy();
            Rg_m[i*4-1]->Draw();
            Rp_m[i*4-1]->Draw("SAME");
        }
        // Plots of parameters versus E and theta
        // --------------------------------------
        mgflux->Divide(2,1);
        mgflux->cd(1);
        P0mg->Draw("COL4");
        mgflux->cd(2);
        P2mg->Draw("COL4");
        mpflux->Divide(2,1);
        mpflux->cd(1);
        P0mp->Draw("COL4");
        mpflux->cd(2);
        P2mp->Draw("COL4");
        egflux->Divide(3,1);
        egflux->cd(1);
        P0eg->Draw("COL4");
        egflux->cd(2);
        P1eg->Draw("COL4");
        egflux->cd(3);
        P2eg->Draw("COL4");
        epflux->Divide(3,1);
        epflux->cd(1);
        P0ep->Draw("COL4");
        epflux->cd(2);
        P1ep->Draw("COL4");
        epflux->cd(3);
        P2ep->Draw("COL4");
#endif
    } // end if check model
    
    // Check that everything is in order
    // ---------------------------------
    if (warnings1+warnings2+warnings3!=0) {
        TerminateAbnormally();
        return 0;
    }

    // Big optimization loop, modifying detector layout
    // ------------------------------------------------
    outfile << "     Starting gradient descent loop " << endl << endl;
    cout    << "     Starting gradient descent loop " << endl << endl;
    double maxUtility = 0.;
    int imax = 0;

    // Einit for FitShowerParams routine
    // ---------------------------------
    for (int ie=0; ie<NEgrid; ie++) {
        Einit[ie] = 0.1*pow(pow(100.,1./NEgrid),ie);
    }

    // beta values for ADAM grad descent
    // ---------------------------------
    for (int ist=0; ist<maxNsteps+1; ist++) {
        powbeta1[ist] = pow(beta1,ist);
        powbeta2[ist] = pow(beta2,ist);
        if (powbeta1[ist]<1./largenumber) powbeta1[ist] = 1./largenumber;
        if (powbeta2[ist]<1./largenumber) powbeta2[ist] = 1./largenumber;
    }

    // Histogram definition
    // --------------------
    double rangex = DetectorSpacing;  // Used if scanU is true 
    double rangey = DetectorSpacing;  // Used if scanU is true
    int NbinsRdistr  = Nunits/5;
    if (NbinsRdistr<40) NbinsRdistr = 40;
    TH1D * U           = new TH1D     ("U",        "Utility function versus epoch", Nepochs, 0.5, (double)Nepochs+0.5);  
    TProfile * Uave    = new TProfile ("Uave",     "", Nepochs/10, 0.5, (double)Nepochs+0.5,0.,100.);  
    TH2D * Layout;  
    TH2D * Showers3;
    TH1D * Rdistr0;
    TH1D * Rdistr;
#ifdef FEWPLOTS 
    TH1D * LLRP;
    TH1D * LLRG;

    // Histograms filled if scanU is true
    // ----------------------------------
    TH2D * Uvsxy       = new TH2D     ("Uvsxy",    "", (int)(sqrt((double)Nepochs)),x[idstar]-rangex,x[idstar]+rangex,                                                     
                                                       (int)(sqrt((double)Nepochs)),y[idstar]-rangey,y[idstar]+rangey);
    TH1D * Uvsx        = new TH1D     ("Uvsx",     "", (int)(sqrt((double)Nepochs)),x[idstar]-rangex,x[idstar]+rangex);                                                     
    TH1D * Uvsy        = new TH1D     ("Uvsy",     "", (int)(sqrt((double)Nepochs)),y[idstar]-rangey,y[idstar]+rangey);                                                     
#endif

#ifdef PLOTS
    TH1D * HEtrue      = new TH1D     ("HEtrue",   "True E spectrum", 9, log(0.01)/log_10, 1.001);
    TH1D * HEmeas      = new TH1D     ("HEmeas",   "Measured E spectrum", 9, log(0.01)/log_10, 1.001);
    TH1D * GFmeas      = new TH1D     ("GFmeas",   "", 5, -3.5, 1.5);
    TH1D * GFtrue      = new TH1D     ("GFtrue",   "", 5, -3.5, 1.5);
    TProfile * dUdx    = new TProfile ("dUdx",     "Derivative dU/dx", Nepochs, -0.5, Nepochs-0.5, 0., 100.);
    TH1D * PosQ        = new TH1D     ("PosQ",     "", Nepochs, -0.5, Nepochs-0.5);
    TH1D * AngQ        = new TH1D     ("AngQ",     "", Nepochs, -0.5, Nepochs-0.5);
    TH1D * EQ          = new TH1D     ("EQ",       "", Nepochs, -0.5, Nepochs-0.5);
    TH2D * LR          = new TH2D     ("LR",       "", Nepochs, -0.5, Nepochs-0.5, 100, -5., 5.);
    TH1D * U_gf        = new TH1D     ("U_gf",     "Relative flux error", Nepochs, -0.5, Nepochs-0.5);
    TH1D * U_ir        = new TH1D     ("U_ir",     "Integrated weighted resolution", Nepochs, -0.5, Nepochs-0.5);
    TH1D * CosDir      = new TH1D     ("CosDir",   "", 100, 0., pi);
    TProfile * CosvsEp = new TProfile ("CosvsEp",  "", Nepochs, -0.5, Nepochs-0.5, 0., pi);
    TH1D * PG          = new TH1D     ("PG","",640,-32.,0.);
    TH1D * PP          = new TH1D     ("PP","",640,-32.,0.);

    // These are filled in FitShowerParams so already declared static
    // ---------------------------------------------------------------
    NumStepsg          = new TProfile ("NumStepsg",  "", 20, 0.1, 10., 0., maxNsteps+1); // to study logLApprox
    NumStepsp          = new TProfile ("NumStepsp",  "", 20, 0.1, 10., 0., maxNsteps+1); // to study logLApprox
#endif

    U->SetMarkerStyle(20);
    U->SetMarkerSize(0.4);
    U->SetMinimum(0.);
    Uave->SetLineColor(kRed);
#ifdef FEWPLOTS
    DE0->SetLineColor(kRed);
    DE->SetLineWidth(3);
    DE0->SetLineWidth(3);
#endif

#ifdef PLOTS
    EQ->SetLineColor(kRed);
    EQ->SetLineWidth(3);
    U_ir->SetLineWidth(3);
    U_ir->SetLineColor(kRed);
    U_gf->SetLineWidth(3);
    PosQ->SetLineWidth(3);
    AngQ->SetLineWidth(3);
    HEtrue->SetLineColor(kRed);
    HEtrue->SetMinimum(0);
    HEtrue->SetLineWidth(3);
    HEmeas->SetLineWidth(3);
    NumStepsp->SetLineColor(kRed);
    PP->SetLineColor(kRed);
#endif


    // SGD stuff
    // ---------
    int epoch = 0;
    double LearningRate[maxUnits];
    double LearningRateC = StartLR; 
    for (int id=0; id<Nunits; id++) {
        LearningRate[id] = StartLR;
    }
    for (int ir=0; ir<NRbins; ir++) {
        LearningRateR[ir] = StartLR;
    }
    double maxDispl = DetectorSpacing;
    if (shape<3) {
        maxDispl = DetectorSpacing;
    } else if (shape==3) {
        maxDispl = SpacingStep;  // max step in R during SGD
    }
    double maxDispl2 = pow(maxDispl,2); 

#ifdef FEWPLOTS
    // Plot histos of residuals in X0, Y0
    // ----------------------------------
    TCanvas * C0; // = new TCanvas ("C0","",500,500);
    // C0->Update();
#endif

#ifdef PLOTS
    TCanvas * C1 = new TCanvas ("C1","",1000,500);
    C1->Divide(4,2);
    C1->cd(1);
    DXP->Draw();
    C1->cd(2);
    DYP->Draw();
    C1->cd(3);
    DXG->Draw();
    C1->cd(4);
    DYG->Draw();
    C1->cd(5);
    DTHP->Draw();
    C1->cd(6);
    DPHP->Draw();
    C1->cd(7);
    DTHG->Draw();
    C1->cd(8);
    DPHG->Draw();
    C1->cd(9);
    DTHPvsT->Draw("COL4");
    C1->cd(10);
    DTHGvsT->Draw("COL4");
    C1->Update();
#endif
    // Print canvas for temporary plots for the first time here
    // --------------------------------------------------------
    TCanvas * CT = new TCanvas ("CT","",1400,650);
    char namepng[120];
#ifdef STANDALONE
    sprintf (namepng,"/lustre/cmswork/dorigo/swgo/MT/Layouts/Layout_Nb=%d_Nu=%d_Ne=%d-%d_Sh=%d_Id=%d_Ep=%d.png", Nbatch, Nunits, startEpoch, startEpoch+Nepochs, shape, indfile, epoch+startEpoch);
#endif
#ifdef INROOT
    sprintf (namepng,"./SWGO/Layouts/Layout_Nb=%d_Nu=%d_Ne=%d-%d_Sh=%d_Id=%d_Ep=%d.png", Nbatch, Nunits, startEpoch, startEpoch+Nepochs, shape, indfile, epoch+startEpoch);
#endif
    CT->Print(namepng);

    // In case we are studying the utility as a function of the position of one detector unit,
    // (scanU = true), we specify here the relevant quantities. The 2d histogram filled is Uvsxy.
    // ------------------------------------------------------------------------------------------
    double x0 = x[idstar];
    double y0 = y[idstar];
    int ind_xincr;
    int ind_yincr;
    int side = (int)(sqrt(Nepochs));

    // Beginning of big optimization loop
    // ----------------------------------
    do { // SGD

        // Define span x and y of generated showers to illuminate layout
        // -------------------------------------------------------------
        double thisr = 0;
        for (int id=0; id<Nunits; id++) {
            thisr = sqrt(x[id]*x[id]+y[id]*y[id]);
            if (thisr>spanR) spanR = thisr; // We never decrease spanR, only increase it if we need to
        }
        // Now we can set the renormalizing factor for the utility
        // -------------------------------------------------------
        ExposureFactor = (spanR+Rslack)/sqrt(Nbatch);

        // Now we can redefine plots that depend on spanR
        // ----------------------------------------------
        Layout         = new TH2D     ("Layout",   "Layout", 500, -spanR-Rslack, spanR+Rslack, 500, -spanR-Rslack, spanR+Rslack);
        Showers3       = new TH2D     ("Showers3", "Showers3",  200, -spanR-Rslack, spanR+Rslack, 200, -spanR-Rslack, spanR+Rslack);
        Rdistr0        = new TH1D     ("Rdistr0",  "R distribution of detectors", NbinsRdistr, 0., spanR+Rslack);
        Rdistr         = new TH1D     ("Rdistr",   "R distribution of detectors", NbinsRdistr, 0., spanR+Rslack);
        NumStepsvsxy   = new TH2D     ("NumStepsvsxy","",20,0.,spanR+Rslack,20,0.1,10.);
        NumStepsvsxyN  = new TH2D     ("NumStepsvsxyN","",20,0.,spanR+Rslack,20,0.1,10.);
        Layout->SetMarkerStyle(20);
        Layout->SetMarkerColor(kBlack);
        Layout->SetMarkerSize(0.4);
        Rdistr0->SetLineColor(kRed);
        Rdistr0->SetLineWidth(3);
        Rdistr->SetLineWidth(3);
        Rdistr->SetMinimum(0);

        // Layout and R distribution are already set
        // -----------------------------------------
        for (int id=0; id<Nunits; id++) {
            Layout->Fill(x[id],y[id]);
            Rdistr0->Fill(sqrt(x[id]*x[id]+y[id]*y[id]));
        }

        // Since we are varying the radius within which we generate showers as we go,
        // we reset the truex0, truey0 of the showers if they are not randomly generated
        // Note that since IsGamma[] indicates a photon for even is, and
        // a proton for odd is (see below when GenerateShower is called),
        // we are alternating photons and protons on the same radii. This
        // must be changed if other geometries are concerned, in case it
        // may interfere with correct placement of detector units.
        // ---------------------------------------------------------------
        if (fixShowerPos) SetShowersXY ();

        // Special runs for checks of gradients will compute these 
        // -------------------------------------------------------
        if (scanU) {
            ind_xincr = epoch%side;
            ind_yincr = epoch/side;
            // Study neighborhood of a detector in terms of U values:
            // Modify the position of this detector to recompute U at different locations
            // --------------------------------------------------------------------------
            x[idstar] = x0 -rangex + 2.*rangex*(0.5+ind_xincr)/side; 
            y[idstar] = y0 -rangey + 2.*rangey*(0.5+ind_yincr)/side;
        }

        // Adjust Learning Rate with scheduling
        // ------------------------------------
        if (commonMode==2) {
            LearningRateC = StartLR * LR_Scheduler(epoch);
            cout    << "     New cycle: Learning rate is now " << LearningRateC << endl;
            outfile << "     New cycle: Learning rate is now " << LearningRateC << endl;
        }

#ifdef PLOTS
        // Reset histograms tracking goodness of position fits and others updated per epoch
        // --------------------------------------------------------------------------------
        DXG->Reset();
        DYG->Reset();
        DXP->Reset();
        DYP->Reset();
        HEtrue->Reset();
        HEmeas->Reset();
        PG->Reset();
        PP->Reset();
#endif
#ifdef FEWPLOTS
        DE->Reset();
#endif
        outfile << endl;
        cout    << endl;
        if (Nthreads==1) {
            outfile << "     Event reconstruction # ";
            cout    << "     Event reconstruction # ";
        }
        Ng_active = 0;
        Np_active = 0;

#ifdef STANDALONE
        // Create multiple threads to generate and reconstruct events
        // ----------------------------------------------------------
        std::vector<std::thread> threads;

        for (int i=0; i<Nthreads; ++i) {
            threads.emplace_back(threadFunction, i);
        }

        // Wait for all threads to finish
        // ------------------------------
        for (auto& thread : threads) {
            thread.join();
        }
#endif
#ifdef INROOT
        threadFunction(0); // there is only one thread, #0
#endif

        // Now collect some info on the showers after possible multithreading on is
        // ------------------------------------------------------------------------
        for (int is=0; is<Nevents+Nbatch; is++) {

            if (is>=Nevents) {
                if (IsGamma[is]) {
                    Ng_active += PActive[is];
                } else {
                    Np_active += PActive[is];
                }
            }

            double Eerror = fabs(Emeas[is][0]-TrueE[is])/TrueE[is];
#ifdef FEWPLOTS
            if (epoch==0 && IsGamma[is] && Active[is]) {
                DE0->Fill(log(TrueE[is])/log_10,Eerror);
            }
#endif

            // Also fill shower 2d distribution with error on position
            // -------------------------------------------------------
            double Derror;
            if (IsGamma[is]) {
                Derror = sqrt(pow(TrueX0[is]-x0meas[is][0],2)+pow(TrueY0[is]-y0meas[is][0],2));
            } else {
                Derror = sqrt(pow(TrueX0[is]-x0meas[is][1],2)+pow(TrueY0[is]-y0meas[is][1],2));
            }
            if (!usetrueXY) {
                Showers3->Fill(TrueX0[is],TrueY0[is],Derror);
            } else {
                Showers3->Fill(TrueX0[is],TrueY0[is],Eerror);
            }
#ifdef PLOTS
            if (IsGamma[is]) {
                if (is<Nevents) {
                    HEtrue->Fill(log(TrueE[is])/log_10);
                } else {
                    HEmeas->Fill(log(Emeas[is][0])/log_10);
                }
            }
#endif
        } // end is loop on all generated events

        // Now events have been declared inactive if they have too bad reconstruction, so we compute the true g fraction
        // -------------------------------------------------------------------------------------------------------------
        if (Ng_active==0.) {
            outfile << "     Sorry, no photon accepted. Terminating." << endl;
            cout    << "     Sorry, no photon accepted. Terminating." << endl;
            warnings3++;
            TerminateAbnormally();
            return 0;
        }
        TrueGammaFraction = Ng_active/(Ng_active+Np_active); // For batch events only

        // Compute stuff that does not depend on detector positions
        // E.g. inverse variance, which is minus the second derivative
        // of the log likelihood versus signal fraction. As the log L is written
        //     log L = Sum_i { MeasFg * Pg + (1-MeasFg)*Pp}, 
        // we get
        //     d^2 logL / dMeasFg^2 = - Sum_i {(Pg-Pp)/(MeasFg*Pg+(1-MeasFg)*Pp)}^2 
        // whence
        //     sigma^2 = 1/Sum_i{...}^2 
        // So we construct the probability density functions Pg[k], Pp[k] for
        // each batch event k, by looping on 1:Nevents and adding Gaussian kernels.
        // ------------------------------------------------------------------------

        // Compute the PDF of the test statistic for all batch showers
        // -----------------------------------------------------------
        double JS = 0.;
        double Pg, Pp;
        for (int k=Nevents; k<Nevents+Nbatch; k++) {
            if (!Active[k]) continue;
            Pg = ComputePDF (k,true);
            Pp = ComputePDF (k,false);
            pg[k] = Pg; // Save a bit of cpu by using non-indexed vars within loop
            pp[k] = Pp;
            double m_x = 0.5*(Pg+Pp);
            if (Pg>0. && Pp>0.) JS += 0.5 * (Pg*log(Pg/m_x)+Pp*log(Pp/m_x));
            // if (MeasFg*Pg+(1.-MeasFg)*Pp>0.) inv_sigmafs2 += pow((Pg-Pp)/(MeasFg*Pg+(1.-MeasFg)*Pp),2);
#ifdef PLOTS
            // Histograms of PDFs:
            // -------------------
            double pgstar = Pg;
            if (pgstar<epsilon) pgstar = epsilon;
            double ppstar = Pp;
            if (ppstar<epsilon) ppstar = epsilon;
            PG->Fill(log(pgstar));
            PP->Fill(log(ppstar));
#endif
        }

        // Compute the gamma fraction in this batch by zeroing the lnL derivative
        // ----------------------------------------------------------------------
        MeasFg = MeasuredGammaFraction(); // Slso computes static inv_sigmafs2

        if (inv_sigmafs2==0.) {
            inv_sigmafs2 = epsilon;
            cout    << "Warning, inv_sigmafs2 = 0" << endl;
            outfile << "Warning, inf_sigmafs2 = 0" << endl;
            warnings1++;
            TerminateAbnormally ();
            return 0;            
        }
        sigmafs2 = 1./inv_sigmafs2;
        MeasFgErr = sqrt(sigmafs2);
        inv_sigmafs  = 1./MeasFgErr; // above we were computing the variance with RCF bound
        //outfile << "     Flux error = " << MeasFgErr << ", JS = " << JS << endl;
        outfile << "     Ng, Np active in this batch = " << Ng_active << " " << Np_active << endl; 
        //cout    << "     Flux error = " << MeasFgErr << ", JS = " << JS << endl;
        cout    << "     Ng, Np active in this batch = " << Ng_active << " " << Np_active << endl; 

        // Construct template of LLRT for gamma and proton, for first ie bin
        // -----------------------------------------------------------------
        double minLRT = -1000.; // was -100000
        /*
        double minLRT = largenumber;
        double maxLRT = -largenumber;
        for (int is=0; is<Nevents; is++) {
            double sqm = sigmaLRT[0][is];
            if (logLRT[0][is]-2.*sqm<minLRT) minLRT = logLRT[0][is]-2.*sqm;
            if (logLRT[0][is]+2.*sqm>maxLRT) maxLRT = logLRT[0][is]+2.*sqm;
        }
        */
#ifdef FEWPLOTS
        LLRP = new TH1D ("LLRP", "", 500, 0., 12.); // log(maxLRT-minLRT+1.)+1); 
        LLRG = new TH1D ("LLRG", "", 500, 0., 12.); // log(maxLRT-minLRT+1.)+1);
        LLRG->SetLineWidth(1);
        LLRP->SetLineWidth(1);
        LLRP->SetLineColor(kRed);
        for (int is=0; is<Nevents; is++) {
            if (!Active[is]) continue;
            // Smooth LLRT distributions with Gaussian kernel
            // by sampling 100 times a Gaussian for every logLRT value
            // -------------------------------------------------------
            for (int irnd=0; irnd<100000/Nevents; irnd++) {
                double thisLRT = logLRT[is]+shift[irnd]*sigmaLRT[is];
                if (is%2==0) { // gamma event
                    if (thisLRT>=minLRT) {
                        LLRG->Fill(log(thisLRT-minLRT+1.));
                    }
                } else { // proton event
                    if (thisLRT>=minLRT) {
                        LLRP->Fill(log(thisLRT-minLRT+1.));
                    }
                }
            }
        }
#endif
        // Below we compute utility function 
        // ---------------------------------
        double U_GF = ComputeUtilityGF();
        double U_IR = ComputeUtilityIR();
        Utility = U_GF + U_IR;
        cout    << " JS div. = " << JS << endl;
        outfile << " JS div. = " << JS << endl;

#ifdef PLOTS
        U_gf->SetBinContent (epoch+1, U_GF);
        U_ir->SetBinContent (epoch+1, U_IR);
#endif
#ifdef FEWPLOTS
        if (scanU) { 
            Uvsxy->Fill(x[idstar],y[idstar],Utility);
            if (fabs(y[idstar]-y0)<rangey/side) Uvsx->Fill(x[idstar],Utility);
            if (fabs(x[idstar]-x0)<rangex/side) Uvsy->Fill(y[idstar],Utility);
            C0 = new TCanvas ("C0","",900,500);
            C0->Divide(3,1);
            C0->cd(1);
            Uvsxy->Draw("COL4");
            C0->cd(2);
            Uvsx->Draw();
            C0->cd(3);
            Uvsy->Draw();
            C0->Update();
            epoch++;
            if (epoch<Nepochs-1) {
                continue; // if we are studying U in neighborhood of detector idstar (see above) we need no gradients
            } else { // at the last cycle we want to compute the dudx, dudy that detector idstar would get
                x[idstar] = x0;
                y[idstar] = y0;
            }
        }
        // if (Utility<0.) Utility = 0.;
        // if (Utility>MaxUtility && epoch>0) Utility = U->GetBinContent(epoch); // use previous value to avoid messing up the U graph
#endif
        U->SetBinContent(epoch+1,Utility);
        Uave->Fill(epoch+1,Utility);
        if (Utility>maxUtility) {
            maxUtility = Utility;
            imax = epoch;
        }
        U->SetMaximum(1.1*maxUtility);

        // Zero a few arrays
        // -----------------
        double aveDR   = 0.; // Keep track of average displacement at each epoch
        double avedUdx = 0.;
        double avedUdy = 0.;
        double displ[maxRbins];
        int Ndispl[maxRbins];
        double prev_displ[maxRbins];
        for (int ir=0; ir<NRbins; ir++) {
            displ[ir]      = 0.;
            prev_displ[ir] = 0.;
            Ndispl[ir]     = 0;
        }
        double commondx = 0;
        double commondy = 0;
        double Momentum_coeff = 0.05; // to be optimized
        double CosThetaEff; // effective angle for successive displacements, used to update learning rate

        // Loop on detector units, to update detector positions following gradient of utility
        // ----------------------------------------------------------------------------------
        if (Nthreads==1) cout << "     Loop on detector # ";

        // Now get derivatives of U vs dx, dy for each detector, by splitting the job in threads.
        // NB displ[], ndispl[], commondx, commondy may cause conflicts between threads. Use
        // multithreading only with commonMode = 0 for the time being.
        // --------------------------------------------------------------------------------------
#ifdef STANDALONE
        // Create multiple threads
        // -----------------------
        std::vector<std::thread> threads2;

        for (int i=0; i<Nthreads; ++i) {
            threads2.emplace_back(threadFunction2, i);
        }

        // Wait for all threads to finish
        // ------------------------------
        for (auto& thread : threads2) {
            thread.join();
        }
#endif
#ifdef INROOT
        threadFunction2 (0); // there is only one thread, #0
#endif
        if (Nthreads==1) {
            cout << endl;
        }

        // We could not update the positions as we went, because we were accumulating global increments. We do it now.
        // -----------------------------------------------------------------------------------------------------------
        if (commonMode==0) { // do not vary R; vary independently x and y

            // Compute average displacement before equalization
            // ------------------------------------------------
            double sumdr2 = 0.;
            for (int id=0; id<Nunits; id++) {
                double dx = dU_dxi[id];
                double dy = dU_dyi[id];
                sumdr2 += dx*dx+dy*dy;
            }
            double multiplier = 1.;
            if (sumdr2>0.) multiplier = 0.3*maxDispl/(sqrt(sumdr2)/Nunits); // calibrate movement

            for (int id=0; id<Nunits; id++) {

                if (scanU && id!=idstar) continue;

                // Update independently each detector position based on gradient of U, ignoring 
                // the symmetry of the problem
                // ----------------------------------------------------------------------------
                double f = LearningRate[id] * LR_Scheduler(epoch);
                double dx = dU_dxi[id] * multiplier * f;
                double dy = dU_dyi[id] * multiplier * f;
                // cout << " id = " << id << " dudx,dudy = " << dU_dxi[id] << "," << dU_dyi[id] 
                //     << " mult = " << multiplier << " dx,dy = " << dx << "," << dy << endl;
                double dr2 = dx*dx+dy*dy;
                if (dr2>maxDispl2) {
                    dx = dx*maxDispl2/dr2;
                    dy = dy*maxDispl2/dr2;
                    // Novfdispl++;
                }
                avedUdx += fabs(dU_dxi[id]);
                avedUdy += fabs(dU_dyi[id]);
                //if (dU_dxi>1. || dU_dyi>1.) {
                    // cout << " id= " << id << " dudx = " << dU_dxi << " dudy = " << dU_dyi 
                    // << " x,x' =" << x[id] << "," << x[id]+dx << " y,y' = " << y[id] << "," << y[id]+dy << endl;
                //}

                // Accumulate information on how consistent are the movements of detectors
                // -----------------------------------------------------------------------
                if (epoch>0) CosThetaEff = ((x[id]-xprev[id])*dx + (y[id]-yprev[id])*dy )/
                                            (sqrt(pow(x[id]-xprev[id],2)+pow(y[id]-yprev[id],2)+epsilon)*sqrt(pow(dx,2)+pow(dy,2)+epsilon));

                // Ok, now update the positions
                // ----------------------------
                xprev[id] = x[id];
                yprev[id] = y[id];
                x[id]     = x[id] + dx;
                y[id]     = y[id] + dy;
                //cout << " id = " << id << "LR = " << LearningRate[id] << " x, dx = " << xprev[id] << " " << dx << " y, yprev = " << yprev[id] << " " << dy << " costh = " << CosThetaEff << endl;

                //cout << id << " " << x[id] << " " << dx << " " << y[id] << " " << dy << endl;
                double maxR = sqrt(pow(x[id],2)+pow(y[id],2));
                if (maxR>spanR+Rslack) {
                    x[id] = x[id]*(spanR+Rslack-epsilon)/maxR;
                    y[id] = y[id]*(spanR+Rslack-epsilon)/maxR;
                }
                
                // Update learning rate based on "costhetaeff" value (see above) - this is the cosine
                // of the angle between the current and the previous detector displacement. If positive,
                // we increase the LR for that unit; if negative, we decrease it.
                // -------------------------------------------------------------------------------------
                double rate_modifier = CosThetaEff; // If using an average one, it might be better to instead use the following definition: 
                                                    // rate_modifier = -1.+2.*pow(0.5*(CosThetaEff+1.),2); 
                                                    // the above function is -1 for x=-1, +1 for x=1, and -0.3 for x=0
                LearningRate[id] *= exp(Momentum_coeff*rate_modifier);
                // Clamp them - we do not want too much variation
                if (LearningRate[id]<MinLearningRate) LearningRate[id] = MinLearningRate;
                if (LearningRate[id]>MaxLearningRate) LearningRate[id] = MaxLearningRate;
#ifdef PLOTS
                if (epoch>0) {
                    double ac = acos(CosThetaEff);
                    CosDir->Fill(ac);
                    CosvsEp->Fill(epoch,ac);
                }
                //LR->Fill(epoch,log(LR_Scheduler(epoch)));
                LR->Fill(epoch,log(LearningRate[id]));
#endif
                aveDR += sqrt(pow(dx,2)+pow(dy,2));
                // cout << "     ID = " << id << ": x+dx,y+dy = " << x[id];
                // if (dx>=0.) cout << "+";
                // cout  << dx << "," << y[id];
                // if (dy>=0.) cout << "+";
                // cout << dy << endl;
                if (scanU) {
                    cout << "xprev, x, dudx = " << xprev[idstar] << " " << x[idstar] << " " << dU_dxi[idstar] 
                         << " yprev, y, dudy = " << yprev[idstar] << " " << y[idstar] << " " << dU_dyi[idstar] << endl;
                } 
            } // end id loop
        } else if (commonMode==1) { // vary R of detectors

            // If cM=1,2 we accumulate average displacements in radius or offset
            // -----------------------------------------------------------------

            // Now we know how the utility varies as a function of the distance of detector i from the showers,
            // measured in terms of the position of the detector x[], y[]. We use this information to vary the
            // detector position by taking all detectors at the same radius and averaging the derivative.
            // ------------------------------------------------------------------------------------------------
            for (int id=0; id<Nunits; id++) {
                double xi = x[id];
                double yi = y[id];
                double Ri; 
                int ir; 
                double dU_dRi;
                double d2 = pow(xi,2)+pow(yi,2);
                if (d2>0.) {
                    Ri = sqrt(d2);
                    double costheta = xi/(Ri+epsilon);
                    double sintheta = yi/(Ri+epsilon);
                    ir = (int)(Ri/(spanR+Rslack)*NRbins);
                    dU_dRi = dU_dxi[id]*costheta + dU_dyi[id]*sintheta;
                } else {
                    dU_dRi = sqrt(pow(dU_dxi[id],2)+pow(dU_dyi[id],2));
                    ir = 0;
                }
                if (ir<NRbins) {
                    displ[ir] += dU_dRi * LearningRateR[ir];
                    Ndispl[ir]++;
                } else {
                    cout << "Warning, ir out of range" << endl;
                    warnings6++;
                }
                if (debug) cout << "     i=" << id << " ir=" << ir << " Ndispl = " << Ndispl[ir] << " displ = " << displ[ir] << endl;
            }

            // Verify consistency of movements and modify learning rate for this radius
            // ------------------------------------------------------------------------
            for (int ir=0; ir<NRbins; ir++) {
                if (displ[ir]*prev_displ[ir]>0.) {
                    LearningRateR[ir] *= Momentum_coeff; // This will apply to next iteration
                } else {
                    LearningRateR[ir] *= -Momentum_coeff;
                }
            }

            // Compute average displacement as f(r)
            // ------------------------------------
            for (int ir=0; ir<NRbins; ir++) {
                prev_displ[ir] = displ[ir];
                if (Ndispl[ir]>0) displ[ir] = displ[ir]/Ndispl[ir];
                if (displ[ir]>maxDispl)  displ[ir] = maxDispl;
                if (displ[ir]<-maxDispl) displ[ir] = -maxDispl;
                if (debug) cout << ir << " " << Ndispl[ir] << " " << displ[ir] << endl;
            }

            // Now we have the required average displacement as a function of R and we apply to detectors
            // ------------------------------------------------------------------------------------------
            for (int id=0; id<Nunits; id++) {            
                double d2 = pow(x[id],2)+pow(y[id],2);
                double dx = 0.;
                double dy = 0.;
                double costh = 0.;
                double sinth = 0.;
                if (d2>0.) {
                    double R = sqrt(d2);
                    int ir = (int)(R/(spanR+Rslack)*NRbins);
                    costh = x[id]/(R+epsilon);
                    sinth = y[id]/(R+epsilon);
                    if (ir<NRbins) {
                        dx = costh * displ[ir];
                        dy = sinth * displ[ir];
                    }
                    x[id] = x[id] + dx;
                    y[id] = y[id] + dy;
                }
                double Rfinal = sqrt(pow(x[id],2)+pow(y[id],2));
                if (Rfinal>spanR+Rslack) {
                    x[id] = (spanR+Rslack-epsilon)*costh;
                    y[id] = (spanR+Rslack-epsilon)*sinth;
                }
                d2 = pow(dx,2)+pow(dy,2);
                if (d2>0.) aveDR += sqrt(d2);     
                if (debug) cout << "aveDR " << aveDR  << " dx,dy " << dx << " " << dy << endl;
            } // end i loop on dets

        } else if (commonMode==2) { // vary all coordinates jointly along common gradient

            for (int id=0; id<Nunits; id++) {
                commondx += dU_dxi[id];
                commondy += dU_dyi[id];
            }

            cout    << "commondx,dy = " << commondx << " " << commondy << endl;
            outfile << "commondx,dy = " << commondx << " " << commondy << endl;
            commondx = LearningRateC*commondx/Nunits;
            commondy = LearningRateC*commondy/Nunits;
            double dr2 = commondx*commondx+commondy*commondy;
            if (dr2>maxDispl2) {
                commondx = commondx * maxDispl2/dr2;
                commondy = commondy * maxDispl2/dr2;
            }
            aveDR = sqrt(pow(commondx,2)+pow(commondy,2))*Nunits;
            for (int id=0; id<Nunits; id++) {
                x[id] += commondx;
                y[id] += commondy;
                if (x[id]>=spanR+Rslack)  x[id] = spanR+Rslack-epsilon;
                if (x[id]<=-spanR-Rslack) x[id] = -spanR-Rslack+epsilon;
                if (y[id]>=spanR+Rslack)  y[id] = spanR+Rslack-epsilon;
                if (y[id]<=-spanR-Rslack) y[id] = -spanR-Rslack+epsilon;
            }
        } // end if commonMode =1 or 2

        cout << "     Epoch = " << epoch 
             << "  Utility value = " << Utility <<  " aveDR = " << aveDR/Nunits;
        if (commonMode==0) cout << " avedU = " << avedUdx/Nunits << "," << avedUdy/Nunits << " Exposure = " << ExposureFactor;
        cout << endl;
        outfile << "     Epoch = " << epoch 
                << "  Utility value = " << Utility << " aveDR = " << aveDR/Nunits;
        if (commonMode==0) outfile << " avedU = " << avedUdx/Nunits << "," << avedUdy/Nunits << " Exposure = " << ExposureFactor;
        outfile << endl;
        for (int id=0; id<Nunits; id++) {
            Layout->Fill(x[id],y[id]);
#ifdef FEWPLOTS
            Rdistr->Fill(sqrt(x[id]*x[id]+y[id]*y[id]));
#endif
        }

#ifdef FEWPLOTS
        // Ensure the Rdistribution histograms stays visible
        // -------------------------------------------------
        int hmax = 0;
        for (int ib=0; ib<NbinsRdistr; ib++) {
            int h = Rdistr->GetBinContent(ib+1);
            if (h>hmax) hmax = h;
            h = Rdistr0->GetBinContent(ib+1);
            if (h>hmax) hmax = h;
        }
        Rdistr0->SetMaximum(hmax*1.1);
#endif

        // Compute agreement metric
        // ------------------------
#ifdef PLOTS
        double QP = 0.5*sqrt(pow(DXP->GetRMS(),2)+pow(DYP->GetRMS(),2)+pow(DXG->GetRMS(),2)+pow(DYG->GetRMS(),2));
        double QA = 0.5*sqrt(pow(DTHG->GetRMS(),2)+pow(DTHP->GetRMS(),2)+pow(DPHG->GetRMS(),2)+pow(DPHP->GetRMS(),2));
        double QE = DEG->GetMean(); // 0.5*sqrt(pow(DEG->GetRMS(),2)+pow(DEP->GetRMS(),2));
        double chi2 = 0.;
        double cumt = 0;
        double cumm = 0;
        double dmax = 0.;
        double sumt = HEtrue->GetEntries();
        double summ = HEmeas->GetEntries();
        for (int i=4; i<=9; i++) {
            int N_t = HEtrue->GetBinContent(i);
            int N_m = HEmeas->GetBinContent(i);
            cumt += 1.*N_t/sumt;
            cumm += 1.*N_m/summ;
            if (dmax<fabs(cumt-cumm)) dmax = fabs(cumt-cumm);
            if (N_t>0) chi2 += pow(1.*(N_t-N_m)/N_t,2.);
        }
        chi2 = chi2 / 6.;
        cout    << "     Performance metric of shower position likelihood = " << QP << " " << QA << " " << QE << " " 
                << chi2 << " " << dmax << " avg steps = " << 1.*NumAvgSteps/DenAvgSteps << endl;
        outfile << "     Performance metric of shower position likelihood = " << QP << " " << QA << " " << QE << " " 
                << chi2 << " " << dmax << " avg steps = " << 1.*NumAvgSteps/DenAvgSteps << endl;
        PosQ->Fill(epoch,QP);
        AngQ->Fill(epoch,QA);
        EQ->Fill(epoch,QE*100.);

        // Current distances plot
        // ----------------------
        C1 = new TCanvas ("C1","",1000,500);
        C1->Divide(5,2);
        C1->cd(1);
        DXP->Draw();
        C1->cd(2);
        DYP->Draw();
        C1->cd(3);
        DXG->Draw();
        C1->cd(4);
        DYG->Draw();
        C1->cd(5);
        DTHP->Draw();
        C1->cd(6);
        DPHP->Draw();
        C1->cd(7);
        DTHG->Draw();
        C1->cd(8);
        DPHG->Draw();
        C1->cd(9);
        DTHPvsT->Draw("COL4");
        C1->cd(10);
        DTHGvsT->Draw("COL4");
        C1->Update();
#endif

        // Summary plot
        // ------------
        CT = new TCanvas ("CT","",1400,850);
        CT->Divide(4,3);
        CT->cd(1);
        if (epoch>0) U->Fit("pol1","Q");
        U->Draw("P");
        Uave->Draw("SAME");
        CT->cd(2);
        Showers3->Draw("COL4");
        Layout->Draw("PSAME");
#ifdef FEWPLOTS
        CT->cd(3);
        Rdistr0->Draw();
        Rdistr->Draw("SAME");
#endif
        CT->cd(4);
        SvsS->SetMarkerStyle(20);
        SvsS->SetMarkerSize(0.3);
        SvsS->Draw("P");
        SvsSP->SetLineColor(kRed);
        SvsSP->SetLineWidth(3);
        SvsSP->Draw("PESAME");
#ifdef PLOTS
        CT->cd(4);
        PosQ->Draw();
        CT->cd(5);
        HEtrue->Draw();
        HEmeas->Draw("SAME");
        CT->cd(9);
        CosDir->Draw();
        CT->cd(10);
        LR->Draw();
        CT->cd(11);
        CosvsEp->Draw();
        CT->cd(12);
        double tmp;
        double maxh = -largenumber;
        for (int i=1; i<=epoch+1; i++) {
            tmp = U_gf->GetBinContent(i);
            if (tmp>maxh) maxh = tmp;
            tmp = U_ir->GetBinContent(i);
            if (tmp>maxh) maxh = tmp;
        }
        U_gf->SetMinimum(0.);
        U_gf->SetMaximum(maxh+0.1*maxh);
        U_gf->Draw();
        U_ir->Draw("SAME");
#endif
#ifdef FEWPLOTS
        CT->cd(6);
        CT->GetPad(6)->SetLogy();
        // LR->Draw();
        DE0->Draw();
        DE->Draw("SAME");
        //EQ->Draw();
        //CT->cd(7);
        // JSSum->Draw();
        //logLvsdr->Draw();
        //PG->Draw();
        //PP->Draw("SAME");
        CT->cd(7);
        CT->GetPad(7)->SetLogy();
        LLRG->Draw();
        LLRP->Draw("SAME");
        CT->cd(8);
        //NumStepsg->Draw();
        //NumStepsp->Draw("SAME");
        NumStepsvsxy->Draw("COL4");
#endif
        CT->Update();
        char namepng[120];
#ifdef STANDALONE
        sprintf (namepng, "/lustre/cmswork/dorigo/swgo/MT/Layouts/Layout_Nb=%d_Nu=%d_Ne=%d-%d_Sh=%d_Id=%d_Ep=%d.png", Nbatch, Nunits, startEpoch, startEpoch+Nepochs, shape, indfile, startEpoch+epoch+1);
#endif
#ifdef INROOT
        sprintf (namepng, "./SWGO/Layouts/Layout_Nb=%d_Nu=%d_Ne=%d-%d_Sh=%d_Id=%d_Ep=%d.png", Nbatch, Nunits, startEpoch, startEpoch+Nepochs, shape, indfile, startEpoch+epoch+1);
#endif
        CT->Print(namepng);

        // Adjust too small learning rates here
        // ------------------------------------
        //if (aveDR<maxDispl/10.) LearningRate = LearningRate*1.5;


        // Debug: check value of utility for same generated batch, after coordinates update
        // --------------------------------------------------------------------------------
        double OldUtility = Utility;
        int iter = 0;
        if (checkUtility && epoch>0) {
            do {
                Ng_active = 0;
                Np_active = 0;
                for (int is=0; is<Nevents+Nbatch; is++) {
                    logLRT[is]   = 0.;
                    sigmaLRT[is] = 1.;
                    if (is%2==0) {
                        IsGamma[is] = true;
                    } else {
                        IsGamma[is] = false;
                    }
                    GenerateShower (is);
                    if (!Active[is]) continue;
                    FindLogLR (is); // Fills logLRT[] array and sigmaLRT
                    if (is>=Nevents) {
                        if (IsGamma[is]) {
                            Ng_active += PActive[is];
                        } else {
                            Np_active += PActive[is];
                        }
                    }
                } // end is loop

                // Compute the PDF of the test statistic for all batch showers
                // -----------------------------------------------------------
                double Pg, Pp;
                for (int k=Nevents; k<Nevents+Nbatch; k++) {
                    if (!Active[k]) continue;
                    Pg = ComputePDF (k,true);
                    Pp = ComputePDF (k,false);
                    pg[k] = Pg; // Save a bit of cpu by using non-indexed vars within loop
                    pp[k] = Pp;
                }

                MeasFg = MeasuredGammaFraction (); // Slso computes static inv_sigmafs2
                if (inv_sigmafs2==0.) {
                    inv_sigmafs2 = epsilon;
                    cout    << "Warning, inv_sigmafs2 = 0" << endl;
                    outfile << "Warning, inf_sigmafs2 = 0" << endl;
                    warnings1++;
                    TerminateAbnormally ();
                    return 0;            
                }
                sigmafs2 = 1./inv_sigmafs2;
                MeasFgErr = sqrt(sigmafs2);
                inv_sigmafs  = 1./MeasFgErr; // above we were computing the variance with RCF bound
                cout << endl;
                Utility = ComputeUtilityGF() + ComputeUtilityIR(); 
                cout << " New vs Old U = " << Utility << " " << OldUtility;
                if (Utility<OldUtility) {
                    for (int id=0; id<Nunits; id++) {
                        x[id] = 0.5*x[id]+0.5*xprev[id];
                        y[id] = 0.5*y[id]+0.5*yprev[id];
                    }
                    cout << " - halving displacements" << endl;
                    iter++;
                } else {
                    cout << " - moving on" << endl;
                }
            } while (Utility<OldUtility && iter<5);
        }

        // Check that everything is in order
        // ---------------------------------
        if (warnings1+warnings2+warnings3!=0) {
            TerminateAbnormally ();
            // SaveLayout();
            return 0;
        }

        // New epoch coming
        // ----------------
        epoch++;

    } while (epoch<Nepochs); // end SGD loop
    // -------------------------------------
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Plot histos of residuals in X0, Y0
    // ----------------------------------
    /*TCanvas * C1 = new TCanvas ("C1","",1200,500);
    C1->Divide(5,2);
    C1->cd(1);
    DXP->Draw();
    C1->cd(2);
    DYP->Draw();
    C1->cd(3);
    DXG->Draw();
    C1->cd(4);
    DYG->Draw();
    C1->cd(5);
    DTHP->Draw();
    C1->cd(6);
    DPHP->Draw();
    C1->cd(7);
    DTHG->Draw();
    C1->cd(8);
    DPHG->Draw();
    C1->cd(9);
    DTHPvsT->Draw("COL4");
    C1->cd(10);
    DTHGvsT->Draw("COL4");
    */

#ifdef PLOTS

    TCanvas * C2 = new TCanvas ("C2","", 1200, 500);
    C2->Divide(3,2);
    C2->cd(1);
    LLRP->Draw();
    LLRG->Draw("SAME");
    C2->cd(2);
    LLRP->SetLineWidth(3);
    LLRP->Draw();
    LLRG->SetLineWidth(3);
    LLRP->SetLineColor(kRed);
    LLRG->Draw("SAME");
    C2->cd(3);
    SigLRT->Draw();
    C2->cd(4);
    SigLvsDRg->Draw("BOX");
    SigLvsDRp->SetLineColor(kRed);
    SigLvsDRp->Draw("SAMEBOX");
    C2->cd(5);
    NmuvsSh->Draw("BOX");
    C2->cd(6);
    NevsSh->Draw("BOX");
 
#endif

    // Plot results
    // ------------
    TCanvas * C = new TCanvas ("C","",1200,800);
    C->Divide(4,2);
    C->cd(1);
    U->SetLineWidth(3);
    U->SetMinimum(0.);
    U->Draw();
    Uave->SetLineColor(kRed);
    Uave->Draw("SAME");
    C->cd(2);
    Showers3->Draw("COL4");
    Layout->Draw("PSAME");
#ifdef FEWPLOTS
    C->cd(3);
    Rdistr->SetLineWidth(3);
    Rdistr->Draw();
    Rdistr0->Draw("SAME");
#endif
#ifdef PLOTS
    C->cd(4);
    dUdx->Draw("PE");
    C->cd(5);
    PosQ->Draw("");
    //AngQ->Draw("SAME");
#endif
#ifdef FEWPLOTS
    C->cd(6);
    for (int i=1; i<=20; i++) {
        for (int j=1; j<=20; j++) {
            NumStepsvsxy->SetBinContent(i,j,NumStepsvsxy->GetBinContent(i,j)/NumStepsvsxyN->GetBinContent(i,j));
        }
    }
    NumStepsvsxy->Draw("COL4");
#endif
#ifdef PLOTS
    C->cd(7);
    PG->Draw();
    PP->Draw("SAME");
    C->cd(8);
    NumStepsg->Draw();
    NumStepsp->Draw("SAME");
#endif
#ifdef FEWPLOTS
    if (scanU) {
        C0 = new TCanvas ("C0","",900,500);
        C0->Divide(3,1);
        C0->cd(1);
        Uvsxy->Draw("COL4");
        C0->cd(2);
        Uvsx->Draw();
        C0->cd(3);
        Uvsy->Draw();
        C0->Update();
    }
#endif

#ifdef PLOTS
    // Plot the distributions of fluxes per m^2
    // ----------------------------------------
    if (plotdistribs) {
        TH1D * MFG3 = new TH1D ("MFG3", "", 150, 0., 1500.);
        TH1D * EFG3 = new TH1D ("EFG3", "", 150, 0., 1500.);
        TH1D * MFP3 = new TH1D ("MFP3", "", 150, 0., 1500.);
        TH1D * EFP3 = new TH1D ("EFP3", "", 150, 0., 1500.);
        TH1D * MFG4 = new TH1D ("MFG4", "", 150, 0., 1500.);
        TH1D * EFG4 = new TH1D ("EFG4", "", 150, 0., 1500.);
        TH1D * MFP4 = new TH1D ("MFP4", "", 150, 0., 1500.);
        TH1D * EFP4 = new TH1D ("EFP4", "", 150, 0., 1500.);
        TH1D * MFG5 = new TH1D ("MFG5", "", 150, 0., 1500.);
        TH1D * EFG5 = new TH1D ("EFG5", "", 150, 0., 1500.);
        TH1D * MFP5 = new TH1D ("MFP5", "", 150, 0., 1500.);
        TH1D * EFP5 = new TH1D ("EFP5", "", 150, 0., 1500.);

        for (int i=0; i<150; i++) {
            double r = i*10.+0.5;
            MFG3->SetBinContent(i+1,MFromG(0.1,0,r,0)/TankArea);
            EFG3->SetBinContent(i+1,EFromG(0.1,0,r,0)/TankArea);
            MFP3->SetBinContent(i+1,MFromP(0.1,0,r,0)/TankArea);
            EFP3->SetBinContent(i+1,EFromP(0.1,0,r,0)/TankArea);
            MFG4->SetBinContent(i+1,MFromG(1.,0,r,0)/TankArea);
            EFG4->SetBinContent(i+1,EFromG(1.,0,r,0)/TankArea);
            MFP4->SetBinContent(i+1,MFromP(1.,0,r,0)/TankArea);
            EFP4->SetBinContent(i+1,EFromP(1.,0,r,0)/TankArea);
            MFG5->SetBinContent(i+1,MFromG(10.,0,r,0)/TankArea);
            EFG5->SetBinContent(i+1,EFromG(10.,0,r,0)/TankArea);
            MFP5->SetBinContent(i+1,MFromP(10.,0,r,0)/TankArea);
            EFP5->SetBinContent(i+1,EFromP(10.,0,r,0)/TankArea);
        }
        EFP3->SetMinimum(0.00000001);
        EFP3->SetMaximum(10000.);
        MFP3->SetMinimum(0.0000001);
        MFP3->SetMaximum(10.);
        EFG3->SetMinimum(0.00000001);
        EFG3->SetMaximum(10000.);
        MFG3->SetMinimum(0.00000001);
        MFG3->SetMaximum(1.);

        TCanvas * G = new TCanvas ("G","", 800, 800);
        G->Divide(2,2);
        G->cd(4);
        MFG3->Draw();
        MFG4->Draw("SAME");
        MFG5->Draw("SAME");
        G->cd(3);
        EFG3->Draw();
        EFG4->Draw("SAME");
        EFG5->Draw("SAME");
        G->cd(2);
        MFP3->Draw();
        MFP4->Draw("SAME");
        MFP5->Draw("SAME");
        G->cd(1);
        EFP3->Draw();
        EFP4->Draw("SAME");
        EFP5->Draw("SAME");
    }    
#endif

    // Write selected histograms to root file
    // --------------------------------------
#ifdef STANDALONE
    string rootPath = "/lustre/cmswork/dorigo/swgo/MT/Root/";
#endif
#ifdef INROOT
    string rootPath = "./SWGO/Root/";
#endif
    std::stringstream rootstr;
    char rnum[100];
    sprintf (rnum,"Nb=%d_Nu=%d_Ne=%d-%d_Sh=%d_Id=%d", Nbatch, Nunits, startEpoch, startEpoch+Nepochs, shape, indfile);
    rootstr << "Swgolo59_";
    string namerootfile = rootPath  + rootstr.str() + num + ".root";
    TFile * rootfile = new TFile (namerootfile.c_str(),"RECREATE");
    rootfile->cd();
    // TCanvas
    if (scanU) C0->Write();
    CT->Write();
#ifdef PLOTS
    C1->Write();
    C2->Write();
    C->Write();
/*    if (checkmodel) {
        TMPe0->Write();
        TMPe1->Write();
        TMPe2->Write();
        TMPe3->Write();
        TMPm0->Write();
        TMPm1->Write();
        TMPm2->Write();
        TMPm3->Write();
        mgflux->Write();
        mpflux->Write();
        egflux->Write();
        epflux->Write();
    }*/
    rootfile->Close();
#endif

    // If requested, write output geometry to file
    // -------------------------------------------
    if (writeGeom) SaveLayout();

    // End of program
    // --------------
    cout    << endl;
    cout    << "     The program terminated correctly. " << endl;
    cout    << "     Warnings: " << endl;
    cout    << "     1 - " << warnings1 << endl;
    cout    << "     2 - " << warnings2 << endl;
    cout    << "     3 - " << warnings3 << endl;
    cout    << "     4 - " << warnings4 << endl;
    cout    << "     5 - " << warnings5 << endl;
    cout    << "     6 - " << warnings6 << endl;
    cout    << "     *****************************************************************" << endl;
    cout    << endl;
    
    // Close dump file
    // ---------------
    outfile << endl;
    outfile << "     The program terminated correctly. " << endl;
    outfile << "     Warnings: " << endl;
    outfile << "     1 - " << warnings1 << endl;
    outfile << "     2 - " << warnings2 << endl;
    outfile << "     3 - " << warnings3 << endl;
    outfile << "     4 - " << warnings4 << endl;
    outfile << "     5 - " << warnings5 << endl;
    outfile << "     6 - " << warnings6 << endl;
    outfile << "--------------------------------------------------------------" << endl;
    outfile.close();

    return 0;
}



// Function that studies the probability of a shower to pass a trigger threshold on the
// number of detectors seeing >0 particles
// ------------------------------------------------------------------------------------
void CheckProb (int Ntrigger=10, int Nev=1000, int Ntrials=1000, int Nu=100, int sh=3,
                double Spacing=50., double SpacingStep = 50., double Rsl=300.) {

    // Pass parameters:
    // ----------------
    // Nu               = number of detector elements. For radial distr, use 1/7/19/37/61/91/127/169/217/271/331/397/469/547/631/721...
    // Spacing          = initial spacing of tanks
    // SpacingStep      = increase in spacing 
    // shape            = geometry of the initial layout (0=hexagonal, 1=taxi, 2=spiral)

    // UNITS
    // -----
    // position: meters
    // angle:    radians
    // time:     nanoseconds
    // energy:   PeV

    if (Ntrigger>maxNtrigger) {
        cout << "Ntrigger cannot exceed " << maxNtrigger << endl;
        return;
    }

    // Get static values from pass parameters
    // --------------------------------------
    Nunits          = Nu;
    DetectorSpacing = Spacing;
    Rslack          = Rsl;
    shape           = sh;

    gStyle->SetOptStat(0);
    gStyle->SetOptFit(0);

    // Get a sound RN generator
    // ------------------------
    delete gRandom;
    myRNG = new TRandom3();

    // Suppress root warnings
    gROOT->ProcessLine( "gErrorIgnoreLevel = 6001;");
    gROOT->ProcessLine( "gPrintViaErrorHandler = kTRUE;");
 
    // Define the current geometry 
    // ---------------------------
    if (readGeom) {
        ReadLayout ();
    } else {
        DefineLayout(DetectorSpacing,SpacingStep);
    }
    
    // Read in parametrizations of particle fluxes and lookup table
    // ------------------------------------------------------------
    int code = ReadShowers ();
    if (code!=0) {
        cout << "Trouble reading showers, terminating. " << endl;
        return;
    }

    TH1D * DPp[4]; 
    TH2D * PfTvsPfCp[4];
    TH1D * DPg[4]; 
    TH2D * PfTvsPfCg[4];
    char namedp[30];
    for (int ie=0; ie<4; ie++) {
        sprintf (namedp, "DPg%d",ie);
        DPg[ie] = new TH1D(namedp,namedp, 100, -0.1, 0.1);
        sprintf (namedp, "PfTvsPfCg%d",ie);
        PfTvsPfCg[ie] = new TH2D(namedp,namedp, 26, 0., 1.04, 26, 0., 1.04);
        sprintf (namedp, "DPp%d",ie);
        DPp[ie] = new TH1D(namedp,namedp, 100, -0.1, 0.1);
        sprintf (namedp, "PfTvsPfCp%d",ie);
        PfTvsPfCp[ie] = new TH2D(namedp,namedp, 26, 0., 1.04, 26, 0., 1.04);
    }

    // Loop on g, p hypotheses
    // -----------------------
    bool Gamma;
    for (int type=0; type<2; type++) {
        if (type==0) Gamma = true;
        if (type==1) Gamma = false;

        // Loop on 4 energy points
        // -----------------------
        for (int ie=0; ie<4; ie++) {
            double E;
            if (type==0) {
                E = 0.2+pow(ie,2.)*0.3;
            } else {
                E = 0.1+pow(ie,2)*0.1;
            }
            // Shower generation loop
            // ----------------------
            for (int is=0; is<Nev; is++) {
                if (is%100==0) cout << ie << " Event # " << is << endl;
                TrueE[is]     = E;
                TrueTheta[is] = 0.;
                TruePhi[is]   = 0.;
                TrueX0[is]    = myRNG->Uniform(-1000,1000);
                TrueY0[is]    = myRNG->Uniform(-1000,1000);

                double mug, eg, mup, ep;
                // Compute a-priori probability that shower passes trigger
                // -------------------------------------------------------
                SumProbgt1[is] = 0.; // expectation value of number of detectors with >=1 particle seen
                for (int id=0; id<Nunits; id++) {
                    double nptot = 0.; // exp value of particles in detector
                    double R = EffectiveDistance(x[id],y[id],TrueX0[is],TrueY0[is],TrueTheta[is],TruePhi[is],0);
                    if (Gamma) {
                        mug = MFromG(E,0,R,0);
                        eg  = EFromG(E,0,R,0);
                        nptot = mug+eg;
                    } else {
                        mup = MFromP(E,0,R,0);
                        ep  = EFromP(E,0,R,0);
                        nptot = mup+ep; 
                    }
                    SumProbgt1[is] += 1. - exp(-nptot); 
                }
                // Compute P(N>Ntrigger) from approx formula
                // -----------------------------------------
                double ws_is = 1.;
                for (int j=0; j<Ntrigger; j++) {
                    ws_is -= exp(-SumProbgt1[is])*pow(SumProbgt1[is],j)/Factorial(j);
                }

                // Resample particles for shower
                // -----------------------------
                double ProbFromTrials = 0.;
                for (int k=0; k<Ntrials; k++) {
                    int Counts = 0;
                    for (int id=0; id<Nunits; id++) {
                        double R = EffectiveDistance(x[id],y[id],TrueX0[is],TrueY0[is],0,0,0);
                        int nm = 0;
                        int ne = 0;
                        if (Gamma) {
                            mug = MFromG(E,0,R,0);
                            eg  = EFromG(E,0,R,0);
                            if (mug>0.) nm = myRNG->Poisson(mug); // otherwise it remains zero
                            if (eg>0.)  ne = myRNG->Poisson(eg);  // otherwise it remains zero
                        } else {
                            mup = MFromP(E,0,R,0);
                            ep  = EFromP(E,0,R,0);
                            if (mup>0.) nm = myRNG->Poisson(mup); // otherwise it remains zero
                            if (ep>0.)  ne = myRNG->Poisson(ep);  // otherwise it remains zero
                        }
                        if (nm+ne>0) Counts++;
                    } // end loop on units
                    if (Counts>=Ntrigger) ProbFromTrials++;
                } // end loop on trials

                // Compute P(N>Ntrigger) from trials
                // ---------------------------------
                ProbFromTrials /= Ntrials;

                if (type==0) {
                    PfTvsPfCg[ie]->Fill(ProbFromTrials,ws_is);
                    DPg[ie]->Fill(ws_is-ProbFromTrials);
                } else {
                    PfTvsPfCp[ie]->Fill(ProbFromTrials,ws_is);
                    DPp[ie]->Fill(ws_is-ProbFromTrials);
                }
            } // end is cycle
        } // end ie cycle
    } // end type cycle

    /*TCanvas * DPC = new TCanvas ("DPC", "", 700,700);
    DPC->Divide (4,2);
    for (int ie=0; ie<4; ie++) {
        DPC->cd(2*ie+1);
        DPC->GetPad(2*ie+1)->SetLogz();
        PfTvsPfCg[ie]->SetLineColor(kRed);
        PfTvsPfCp[ie]->SetLineColor(kBlue);
        PfTvsPfCg[ie]->Draw("BOX");
        DPC->cd(2*ie+2);
        DPC->GetPad(2*ie+2)->SetLogz();
        PfTvsPfCp[ie]->Draw("BOX");
    }
    */
    TCanvas * DPC2 = new TCanvas ("DPC", "", 700,700);
    DPC2->Divide (4,4);
    for (int ie=0; ie<4; ie++) {
        DPC2->cd(4*ie+1);
        DPC2->GetPad(4*ie+1)->SetLogz();
        PfTvsPfCg[ie]->SetLineWidth(3);
        PfTvsPfCg[ie]->SetLineColor(kRed);
        PfTvsPfCg[ie]->Draw("BOX");
        DPC2->cd(4*ie+2);
        DPg[ie]->SetLineWidth(3);
        DPg[ie]->SetLineColor(kRed);
        DPg[ie]->Draw();
        DPC2->cd(4*ie+3);
        DPC2->GetPad(4*ie+3)->SetLogz();
        PfTvsPfCp[ie]->SetLineWidth(3);
        PfTvsPfCp[ie]->SetLineColor(kBlue);
        PfTvsPfCp[ie]->Draw("BOX");
        DPC2->cd(4*ie+4);
        DPp[ie]->SetLineWidth(3);
        DPp[ie]->SetLineColor(kBlue);
        DPp[ie]->Draw();
    }

    return;
    
}