#define transactions_cxx
#include "transactions.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>

void transactions::Loop()
{
//   In a ROOT session, you can do:
//      root> .L transactions.C
//      root> transactions t
//      root> t.GetEntry(12); // Fill t data members with entry number 12
//      root> t.Show();       // Show values of entry 12
//      root> t.Show(16);     // Read and show values of entry 16
//      root> t.Loop();       // Loop on all entries
//

//     This is the loop skeleton where:
//    jentry is the global entry number in the chain
//    ientry is the entry number in the current Tree
//  Note that the argument to GetEntry must be:
//    jentry for TChain::GetEntry
//    ientry for TTree::GetEntry and TBranch::GetEntry
//
//       To read only selected branches, Insert statements like:
// METHOD1:
    fChain->SetBranchStatus("*",0);  // disable all branches

    fChain->SetBranchStatus("color",1);  // activate branchname
    fChain->SetBranchStatus("mass",1);  // activate branchname
    fChain->SetBranchStatus("n",1);  // activate branchname
// METHOD2: replace line
//    fChain->GetEntry(jentry);       //read all branches
//by  b_branchname->GetEntry(ientry); //read only this branch

// At this point we declare the histograms we need
   TH1F *h1 = new TH1F("orange","orange ",1400,0,14); 
   TH1F *h2 = new TH1F("green","green ",1400,0,14);
   TH1F *h3 = new TH1F("yellow","yellow ",1400,0,14);
   TH1F *h4 = new TH1F("red","red ",1400,0,14);
   TH1F *h5 = new TH1F("white","white ",1400,0,14);
   TH1F *htot = new TH1F("all","all ",1400,0,14);

   if (fChain == 0) return;

   Long64_t nentries = fChain->GetEntriesFast();

   Long64_t nbytes = 0, nb = 0;
   for (Long64_t jentry=0; jentry<nentries;jentry++) 
   {
      Long64_t ientry = LoadTree(jentry);
      if (ientry < 0) break;
      nb = fChain->GetEntry(jentry);   nbytes += nb;
      // if (Cut(ientry) < 0) continue;

      //Analysis progress
      if(jentry%100==0)std::cout << "Analysed a total of: " << jentry << " events" << std::endl;

      
      //ANALYSIS AND HISTROGRAM FILLING

      Double_t mass_max = 0;
      Double_t White_mass = 0;
      Int_t farmer = 0;
      Int_t rich_farmers = 0;
      float mass_tot=0;
      int red_p =0;
      float red_av[nentries];

      
      for(int i=0; i < n; i++)
      {
         Double_t index = 0 ;
         Double_t mass_d = mass[i];
         Double_t color_d = color[i];
         //std::cout<< mass->at() <<std::endl;
         htot->Fill(mass[i]);
         if(color_d==0)h1->Fill(mass[i]);
         if(color_d==1)h2->Fill(mass[i]);
         if(color_d==2)h3->Fill(mass[i]);
         if(color_d==3)h4->Fill(mass[i]); 
         if(color_d==4)h5->Fill(mass[i]);

         if(mass_d > mass_max)
         {
            mass_max=mass_d;
            farmer = jentry;
         }

         if(color_d==4)White_mass += mass[i];
         if(color_d==0 && color_d==1 && color_d==2 && color_d==3 && color_d==4) rich_farmers+=1;

         double pumpkin[13][5];

         mass_tot=0;
         red_p=0;
         if(color_d==3)
         {
            mass_tot +=mass[i];
            red_p +=1;
         }
      
      }

      red_av=mass_tot/red_p;
      std::cout<< "Average Red: " << red_av <<std::endl;

      /* std::cout<< "Biggest Pumpkin: " << mass_max <<std::endl;
      std::cout<< "Owner of Biggest: " << farmer <<std::endl;
      std::cout<< "Total Mass of white pumpkins: " << White_mass <<std::endl;
      std::cout<< "Number of farmers that have all colors of pumpkins: " << rich_farmers <<std::endl; */
   
   
   }


   
   TCanvas c("c", "mass distribution", 1000, 400);
   TLegend *leg = new TLegend(0.7,0.7,0.9,0.9);

   leg->AddEntry(h1, "Orange", "l");
   leg->AddEntry(h2, "Green", "l");
   leg->AddEntry(h3, "Yellow", "l");
   leg->AddEntry(h4, "Red", "l");
   leg->AddEntry(h5, "White", "l");
   leg->AddEntry(htot, "All", "l");

   h1->SetLineColor(2);   
   h2->SetLineColor(3);
   h3->SetLineColor(4);
   h4->SetLineColor(5);
   h5->SetLineColor(6); 
   htot->SetLineColor(1); 

   gStyle->SetOptStat(0);

   h1->GetYaxis()->SetRangeUser(0.,50000);
   h1->GetXaxis()->SetLimits(0.,14.);
   h1->GetYaxis()->SetTitle("number of pumpkins / 50 g");
   h1->GetXaxis()->SetTitle("pumpkins mass [kg]");

   h1->Draw();
   h2->Draw("same");
   h3->Draw("same");
   h4->Draw("same");
   h5->Draw("same");
   htot->Draw("same");
   leg->Draw("same");

   c.Draw();
   c.SaveAs("Mass.png");
   
}

