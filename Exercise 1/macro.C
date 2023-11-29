#include <stdio.h> 
#include <stdlib.h> 
#include <math.h>
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>

void macro(){
    TFile f("pumpkins_big.root");

    TTree * tree = (TTree *) f.Get("transactions");
    TLegend *leg = new TLegend(0.7,0.7,0.9,0.9);

    TLeaf *mass = (TLeaf*)tree->GetLeaf("mass");
    TLeaf *color = (TLeaf*)tree->GetLeaf("color");
    TLeaf *n = (TLeaf*)tree->GetLeaf("n");

    TH1F *h1 = new TH1F("orange","orange ",1400,0,14); 
    TH1F *h2 = new TH1F("green","green ",1400,0,14);
    TH1F *h3 = new TH1F("yellow","yellow ",1400,0,14);
    TH1F *h4 = new TH1F("red","red ",1400,0,14);
    TH1F *h5 = new TH1F("white","white ",1400,0,14);
    TH1F *htot = new TH1F("all","all ",1400,0,14);

    for(int i=0; i<tree->GetEntries(); i++)
    {
        tree->GetEntry(i);

        int np = n->GetValue();

        for(int j=0; j<np; j++)
        {
        double mass_d = mass->GetValue(j);
        double color_d = color->GetValue(j);
        //std::cout<< mass->GetValue() <<std::endl;
        htot->Fill(mass_d);
        if(color_d==0)h1->Fill(mass_d);
        if(color_d==1)h2->Fill(mass_d);
        if(color_d==2)h3->Fill(mass_d);
        if(color_d==3)h4->Fill(mass_d); 
        if(color_d==4)h5->Fill(mass_d);
        }
    }

 
//Second 
    
    /* tree->Draw("mass>>h_o", "color==0");
    tree->Draw("mass>>h_g", "color==1");
    tree->Draw("mass>>h_y", "color==2");
    tree->Draw("mass>>h_r", "color==3");
    tree->Draw("mass>>h_w", "color==4");
    tree->Draw("mass>>h_t");

    TH1F *h1 = (TH1F*)gDirectory->Get("h_o");
    TH1F *h2 = (TH1F*)gDirectory->Get("h_g");
    TH1F *h3 = (TH1F*)gDirectory->Get("h_y");
    TH1F *h4 = (TH1F*)gDirectory->Get("h_r");
    TH1F *h5 = (TH1F*)gDirectory->Get("h_w"); 
    TH1F *htot = (TH1F*)gDirectory->Get("h_t"); */
    

    TCanvas c("c", "mass distribution", 1000, 400);

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
    c.SaveAs("h1.png");
}
