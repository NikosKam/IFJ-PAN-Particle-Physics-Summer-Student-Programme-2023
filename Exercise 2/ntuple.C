#define ntuple_cxx
#include "ntuple.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <iostream>
#include <iomanip>
#include <TChain.h>
#include <TLorentzVector.h>

void ntuple::Loop()
{
//   In a ROOT session, you can do:
//      root> .L ntuple.C
//      root> ntuple t
//      root> t.GetEntry(12); // Fill t data members with entry number 12
//      root> t.Show();       // Show values of entry 12
//      root> t.Show(16);     // Read and show values of entry 16
//      root> t.Loop();       // Loop on all entries
//
   if (fChain == 0) return;

   Long64_t nentries = fChain->GetEntriesFast();

   TFile output(this->outputFileName.data(), "recreate");
   TH1F h_nparticles("h_nparticles", "", 100, 0, 1000);
   TH1F h_mulike("h_mulike", "", 120, 0, 1.2);
   TH1F h_x("h_x", "", 100, -1e-4, 1e-4);
   TH1F h_r("h_r", "", 50, 0, 0.5);
   TH1F h_mass("h_mass", "", 1000, 0, 100);
   TH1F h_pt("h_pt", "", 1000, 0, 100);
   TH1F h_leadingpt("h_leadingpt", "", 1000, 0, 100);
   TH1F h_subleadingpt("h_subleadingpt", "", 1000, 0, 100);
   TH1F h_theta("h_theta", "", 50, 0, 5);
   TH1F h_eta("h_eta", "", 300, -15, 15);
   TH1F h_phi("h_phi", "", 100, -5, 5);
   TH2F h_x_y("h_x_y", "", 100, -1e-4, 1e-4, 100, -1e-4, 1e-4);
   TH2F h_r1_r2("h_r1_r2", "", 100, 0, 0.3, 100, 0, 0.3);


   Long64_t nbytes = 0, nb = 0;
   for (Long64_t jentry=0; jentry<nentries;jentry++) 
   {
      Long64_t ientry = LoadTree(jentry);
      if (ientry < 0) break;
      nb = fChain->GetEntry(jentry);   nbytes += nb;
      // if (Cut(ientry) < 0) continue;
      int nparticles = id->size();
      h_nparticles.Fill(nparticles);
      
      
      for(int particle=0; particle < nparticles; particle++)
      {
        if(charge!=0 && sqrt((px->at(particle))*(px->at(particle))+(py->at(particle))*(py->at(particle)))>2)
        {
          if(abs(id->at(particle))==13 || abs(id->at(particle))>2300)
          {
            h_x.Fill(x->at(particle));
            h_mulike.Fill(mu_like->at(particle));
            h_x_y.Fill(x->at(particle), y->at(particle));

            for(int i=particle+1; i< nparticles; i++)
            {
              if(charge->at(particle)!=charge->at(i))
              {
                TLorentzVector p1, p2;
                p1.SetPxPyPzE(px->at(particle),py->at(particle),pz->at(particle),e->at(particle));
                p2.SetPxPyPzE(px->at(i),py->at(i),pz->at(i),e->at(i));
                double m_inv = (p1+p2).M();

                /* double p1, p2 = 0, 0;
                p1=sqrt(px->at(particle)^2+py->at(particle)^2+pz->at(particle)^2+e->at(particle)^2);
                p2=sqrt(px->at(particle+1)^2+py->at(particle+1)^2+pz->at(particle+1)^2+e->at(particle+1)^2);

                double m_inv = sqrt((px->at(particle)+px->at(particle+1))^2+(py->at(particle)+py->at(particle+1))^2+(pz->at(particle)+pz->at(particle+1))^2+(e->at(particle)+e->at(particle+1))^2);*/              
                h_mass.Fill(m_inv);

                double theta = (p1+p2).Theta();//acos(p1*p2/(((px->at(particle))^2+(py->at(particle))^2+(pz->at(particle))^2+(e->at(particle))^2)*((px->at(i))^2+(py->at(i))^2+(pz->at(i))^2+(e->at(i))^2)));
                h_theta.Fill(theta);

                double eta = (p1+p2).PseudoRapidity();
                h_eta.Fill(eta);

                double phi = (p1+p2).Phi();
                h_phi.Fill(phi);

                double pt = (p1+p2).Pt();
                h_pt.Fill(pt);

                double leadingpt = (p1).Pt();
                h_leadingpt.Fill(leadingpt);

                double subleadingpt = (p2).Pt();
                h_subleadingpt.Fill(subleadingpt);

                double r = sqrt(((x->at(particle))-(x->at(i)))*((x->at(particle))-(x->at(i))) + ((y->at(particle))-(y->at(i)))*((y->at(particle))-(y->at(i))) + ((z->at(particle))-(z->at(i)))*((z->at(particle))-(z->at(i))));
                h_r.Fill(r);


                double r1 = sqrt((x->at(particle))*(x->at(particle)) + (y->at(particle))*(y->at(particle)) + (z->at(particle))*(z->at(particle)));
                double r2 = sqrt((x->at(i))*(x->at(i)) + (y->at(i))*(y->at(i)) + (z->at(i))*(z->at(i)));
                h_r1_r2.Fill(r1, r2);
              }
            }

            
          }
        }
        
      }
      
      
   }
   output.Write();
   output.Close();
}

int main(int argc, char** argv){

  if(argc < 3){
    std::cout << "Usage:\n\t" << argv[0] << " output.root input1.root [input2.root input3.root ...]\n\n";
    return 1;
  }

  std::cout << "Output: " << argv[1] << "\n";
  // TChain is like a TTree, but can work across several root files
  TChain * chain = new TChain("ntuple"); 
  std::cout << "Inputs:\n";
  for(int i=2; i<argc; i++){
    std::cout << "\t" << argv[i] << "\n";
    chain->Add(argv[i]);
  }

  ntuple t(chain);
  t.outputFileName = argv[1];
  t.Loop();

  std::cout << "[ DONE ]\n\n";

}
