// #include "/Users/atrzupek/AtlasStyle.C"
// #include "/Users/atrzupek/AtlasLabels.C"
// #include "/Users/atrzupek/AtlasUtils.h"
 TGraph *GetRatioGraph(TGraph *roc_b1,TGraph *roc_b0);

bool IsNaN(double t);

//void roc_all(TString scaled_b = "PPSS23_nn/pdf_files/ROC_b_vs_lrej_v77_jzincllr0.1_bs2000_mean_squared_logarithmic_error.keras.txt", TString scaled_c = "PPSS23_nn/pdf_files/ROC_c_vs_lrej_v77_jzincllr0.1_bs2000_mean_squared_logarithmic_error.keras.txt", TString unscaled_b = "unscaled/pdf_files/ROC_b_vs_lrej_v77_jzincllr0.1_bs2000_mean_squared_logarithmic_error.keras.txt", TString unscaled_c = "unscaled/pdf_files/ROC_c_vs_lrej_v77_jzincllr0.1_bs2000_mean_squared_logarithmic_error.keras.txt", TString cent="low"){
//void roc_all(TString scaled_b = "PPSS23_nn/pdf_files/ROC_b_vs_lrej_v77_jzincllr0.05_bs2000_categorical_crossentropy.keras.txt", TString scaled_c = "PPSS23_nn/pdf_files/ROC_c_vs_lrej_v77_jzincllr0.05_bs2000_categorical_crossentropy.keras.txt", TString unscaled_b = "unscaled/pdf_files/ROC_b_vs_lrej_v77_jzincllr0.05_bs2000_categorical_crossentropy.keras.txt", TString unscaled_c = "unscaled/pdf_files/ROC_c_vs_lrej_v77_jzincllr0.05_bs2000_categorical_crossentropy.keras.txt", TString cent="low"){
void roc_all(TString scaled_b = "PPSS23_nn/pdf_files/ROC_b_vs_lrej_v77_jzincllr0.1_bs2000_mse.keras.txt", TString scaled_c = "PPSS23_nn/pdf_files/ROC_c_vs_lrej_v77_jzincllr0.1_bs2000_mse.keras.txt", TString unscaled_b = "unscaled/pdf_files/ROC_b_vs_lrej_v77_jzincllr0.1_bs2000_mse.keras.txt", TString unscaled_c = "unscaled/pdf_files/ROC_c_vs_lrej_v77_jzincllr0.1_bs2000_mse.keras.txt", TString cent="low"){

  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  // gROOT->LoadMacro("/Users/atrzupek/AtlasUtils.C");
  // SetAtlasStyle(0.05);

  //  80%      70%       60%      50%       40%      30%       20%      10%    
  double centralityBin[] ={-10.0, 0.063719, 0.14414, 0.289595, 0.525092, 0.87541, 1.36875, 2.04651,  2.98931,  5.5 };
  TString CentTxt[] = {"80-100%","70-80%","60-70%","50-60%","40-50%","30-40%","20-30%","10-20%","0-10%"};
  int cent_low=1,cent_mid=5,cent_hig=8;

  TLegend *ll1 = new TLegend(0.3, 0.7, 0.5, 0.9,NULL,"brNDC");//0.2,0.05,0.5,0.35
  ll1->SetTextSize(0.05);  ll1->SetLineColor(0);  ll1->SetShadowColor(0);

  
  TString ss="2018";
  if(cent=="incl")ss="pp";
  int N=1;
  if(ss=="2015")N=5;
  if(ss=="pp")N=3;

  TCanvas *c1 = new TCanvas("c1","A Simple Graph Example",200,10,400,400);  
  c1->Range(-1.5,-180.2063,1.5,1621.856);
  c1->SetFillColor(0);
  c1->SetBorderMode(0);
  c1->SetBorderSize(2);
  c1->SetFrameBorderMode(0);
  c1->SetFrameBorderMode(0);
  
  // ------------>Primitives in pad: c1_1
  TPad *c1_1 = new TPad("c1_1", "c1_1",0.01,0.31,0.99,0.99);
  c1_1->Draw();
  c1_1->cd();
  c1_1->Range(0.54375,-0.7585808,1.10625,3.949986);
  c1_1->SetFillColor(0);
  c1_1->SetBorderMode(0);
  c1_1->SetBorderSize(2);
  c1_1->SetLogy();
  c1_1->SetFrameBorderMode(0);
  c1_1->SetLeftMargin(0.16);
  c1_1->SetRightMargin(0.05);
  c1_1->SetTopMargin(0.05);
  c1_1->SetBottomMargin(0.001);
  
  c1_1->cd();
  c1_1->SetLogy();
  
  //c1->GetPad(1)->SetMargin(0.25,0.02,0.2,0.05);
  
  TH1F *Graph_Graph01 = new TH1F("Graph_Graph01","Graph_Graph01",100,0.5,1.0);
  Graph_Graph01->SetMinimum(0.2);
  Graph_Graph01->SetMaximum(4000);
  if(ss=="pp")Graph_Graph01->SetMaximum(10000.);
  Graph_Graph01->SetDirectory(0);
  Graph_Graph01->SetStats(0);
  Int_t ci;      // for color index setting
  TColor *color; // for color definition with alpha
  ci = TColor::GetColor("#000099");
  Graph_Graph01->SetLineColor(ci);
  Graph_Graph01->GetXaxis()->SetTitle("b-jet efficiency");
  Graph_Graph01->GetXaxis()->SetLabelFont(42);
  Graph_Graph01->GetXaxis()->SetLabelSize(0.045);
  Graph_Graph01->GetXaxis()->SetTitleSize(0.055);
  Graph_Graph01->GetXaxis()->SetTitleOffset(1);
  Graph_Graph01->GetXaxis()->SetTitleFont(42);
  Graph_Graph01->GetYaxis()->SetTitle(" rejection");
  Graph_Graph01->GetYaxis()->SetTitleOffset(1.);
  Graph_Graph01->GetYaxis()->SetLabelFont(42);
  Graph_Graph01->GetYaxis()->SetLabelSize(0.045);
  Graph_Graph01->GetYaxis()->SetTitleSize(0.055);
  Graph_Graph01->GetYaxis()->SetTitleFont(42);
  Graph_Graph01->Draw();

  TH1F *Graph_Graph03 = new TH1F("Graph_Graph03","Graph_Graph03",100,0.5,1.0);
  Graph_Graph03->SetMinimum(0.2);
  Graph_Graph03->SetMaximum(4000);
  if(ss=="pp")Graph_Graph03->SetMaximum(10000.);
  Graph_Graph03->SetDirectory(0);
  Graph_Graph03->SetStats(0);
  //Int_t ci;      // for color index setting
  //TColor *color; // for color definition with alpha
  //ci = TColor::GetColor("#000099");
  Graph_Graph03->SetLineColor(ci);
  Graph_Graph03->GetXaxis()->SetTitle("b-jet efficiency");
  Graph_Graph03->GetXaxis()->SetLabelFont(42);
  Graph_Graph03->GetXaxis()->SetLabelSize(0.045);
  Graph_Graph03->GetXaxis()->SetTitleSize(0.055);
  Graph_Graph03->GetXaxis()->SetTitleOffset(1);
  Graph_Graph03->GetXaxis()->SetTitleFont(42);
  Graph_Graph03->GetYaxis()->SetTitle(" rejection");
  Graph_Graph03->GetYaxis()->SetTitleOffset(1.);
  Graph_Graph03->GetYaxis()->SetLabelFont(42);
  Graph_Graph03->GetYaxis()->SetLabelSize(0.045);
  Graph_Graph03->GetYaxis()->SetTitleSize(0.055);
  Graph_Graph03->GetYaxis()->SetTitleFont(42);
  Graph_Graph03->Draw();

  TGraph *roc_b[5];	
  TString pt_th="default text";
  TString ver="v";
  TString fhn[4] = {scaled_b , scaled_c, unscaled_b , unscaled_c};   
  for(int ifile=0;ifile<4;ifile++){
    ver="v";
     if(ss=="2015"){
       if(ifile==0){ver="v210";pt_th="default p_{T} cut";}
       if(ifile==1){ver="v211";pt_th="p_{T}> 1 GeV";}
       if(ifile==2){ver="v212";pt_th="p_{T}> 2 GeV";}
       if(ifile==3){ver="v213";pt_th="p_{T}> 3 GeV";}
       if(ifile==4){ver="v214";pt_th="p_{T}> 4 GeV";}
     }
     if(ss=="2018"){
       //if(ifile==0){ver="v97";pt_th="default p_{T} cut";}
       //if(ifile==1){ver="v98";pt_th="p_{T}> 1 GeV";}
       //if(ifile==2){ver="v99";pt_th="p_{T}> 2 GeV";}
       //if(ifile==3){ver="v100";pt_th="p_{T}> 3 GeV";}
       //if(ifile==4){ver="v96";pt_th="p_{T}> 4 GeV";}
       if(ifile==0){ver="v50";pt_th="default p_{T} cut";}
       if(ifile==1){ver="v51";pt_th="p_{T}> 1 GeV";}
       if(ifile==2){ver="v52";pt_th="p_{T}> 2 GeV";}
       if(ifile==3){ver="v53";pt_th="p_{T}> 3 GeV";}
       if(ifile==4){ver="v54";pt_th="p_{T}> 4 GeV";}
     }
//    if(ss=="pp"){
//       if(ifile==0){ver="v160";pt_th="default p_{T} cut";}
//       if(ifile==1){ver="v104";pt_th="p_{T}> 1 GeV";}
//       if(ifile==2){ver="v105";pt_th="p_{T}> 2 GeV";}
//    }
    /* if(ver=="v")continue;
    if(ss!="pp" || ver=="v160"){
      //fhn="../../txt_test/test4_12Ma_ip3d/ROC_";
      fhn="../../txt_test/test4_12Ma/ROC_";
      //fhn="../../txt_test/test4_40M/ROC_";
      if(q==0)fhn+="b";
      if(q==1)fhn+="c";
      if(ver=="v160")
       fhn+="_vs_lrej_"+ver+"_"+"cent0.txt"; 
      else
       fhn+="_vs_lrej_"+ver+"_"+cent+".txt";
    }
    else{
 //ROC_b_vs_lrej_incl.txt
      fhn="../../"+ver+"/ROC_";
      if(q==0)fhn+="b";
      if(q==1)fhn+="c";
      fhn+="_vs_lrej_incl.txt";
    } */
   std :: cout << " input " << ifile << ": "  << fhn[ifile] << std :: endl;

   roc_b[ifile]= new TGraph(fhn[ifile]);	
   roc_b[ifile]->SetFillColor(0);
   roc_b[ifile]->SetFillStyle(1000);
   roc_b[ifile]->SetMarkerStyle(1);
   roc_b[ifile]->SetLineWidth(3);  
   roc_b[ifile]->SetMarkerSize(1.);
   
   if(ifile==0){     
     roc_b[ifile]->SetLineColor(1);
     roc_b[ifile]->SetMarkerColor(1);
     roc_b[ifile]->SetLineStyle(10);//NEW
     roc_b[ifile]->SetHistogram(Graph_Graph01);
     roc_b[ifile]->Draw("AClP");
     ll1->AddEntry(roc_b[ifile],"Scaled light-flavour jet ");
   }
   else if(ifile==1) {
     roc_b[ifile]->SetLineColor(4);
     roc_b[ifile]->SetLineStyle(1);
     roc_b[ifile]->SetMarkerColor(4);
     
     roc_b[ifile]->Draw("samelP");
     ll1->AddEntry(roc_b[ifile],"Scaled c-jet ");
   }
   else if(ifile==2) {
     roc_b[ifile]->SetLineColor(2);
     roc_b[ifile]->SetLineStyle(10);
     roc_b[ifile]->SetMarkerColor(2);
     
     roc_b[ifile]->Draw("samelP");
     ll1->AddEntry(roc_b[ifile],"Unscaled light-flavour jet ");
   }
   else if(ifile==3) {
     roc_b[ifile]->SetLineColor(kCyan);
     roc_b[ifile]->SetLineStyle(1);
     roc_b[ifile]->SetMarkerColor(kCyan);
     roc_b[ifile]->Draw("samelP");
     ll1->AddEntry(roc_b[ifile],"Unscaled c-jet ");
   }
   /* else if(ifile==4) {
     roc_b[ifile]->SetLineColor(3);
     roc_b[ifile]->SetLineStyle(1);
     roc_b[ifile]->SetMarkerColor(3);
     roc_b[ifile]->Draw("samelP");
   } */
   
    
  }
  double xs=0.6; double ys=0.6;
  // ATLASLabel(xs,ys+0.29,"Internal");
  // myText(xs,ys+0.24,1,"Simulation");
  
  // if(ss=="pp"){
  //   myText(xs,ys+0.19,1,"pp, 5.02 TeV");
  // }
  // if(ss=="2015") {
  //   myText(xs,ys+0.19,1,"Pb+Pb 2015, 5.02 TeV");
  // }
  // if(ss=="2018") {
  //   myText(xs,ys+0.19,1,"Pb+Pb 2018, 5.02 TeV");
  // }
  // myText(xs,ys+0.13,1,"jet p_{T}: 50-500 GeV");
  // if(cent=="cent1")myText(xs+0.1,ys+0.01,1,CentTxt[cent_low]);
  // if(cent=="cent5")myText(xs+0.1,ys+0.01,1,CentTxt[cent_mid]);
  // if(cent=="cent8")myText(xs+0.1,ys+0.01,1,CentTxt[cent_hig]);

  // if(ss=="pp"){
  //   if(q==0){
  //     TGraph *roc_b_urej_def= new TGraph("../urej_def.txt");
  //     roc_b_urej_def->SetFillStyle(1000);
  //     roc_b_urej_def->SetMarkerColor(2);
  //     roc_b_urej_def->SetMarkerStyle(1);
  //     roc_b_urej_def->SetLineColor(9);
  //     roc_b_urej_def->SetLineStyle(1);
  //     roc_b_urej_def->SetLineWidth(1);
  //     roc_b_urej_def->SetMarkerSize(1.);
      
  //     roc_b_urej_def->Draw("samel");
  //     ll1->AddEntry(roc_b_urej_def,"from reconstruction");
  //   }
  //   if(q==1){
  //     TGraph *roc_b_crej_def= new TGraph("../../bigpp/crej_def.txt");
  //     roc_b_crej_def->SetFillStyle(1000);
  //     roc_b_crej_def->SetMarkerColor(9);
  //     roc_b_crej_def->SetMarkerStyle(1);
  //     roc_b_crej_def->SetLineColor(9);
  //     roc_b_crej_def->SetLineWidth(1);
  //     roc_b_crej_def->SetMarkerSize(1.);
      
  //     roc_b_crej_def->Draw("samel");
      
  //     ll1->AddEntry(roc_b_crej_def,"from reconstruction");
  //   }
  // }
  ll1->Draw();
  
     
  // ------------>Primitives in pad: c1_2
  c1->cd();
  TPad *c1_2 = new TPad("c1_2", "c1_2",0.01,0.01,0.99,0.29);
  c1_2->Draw();
  c1_2->cd();
  c1_2->SetFillColor(0);
  c1_2->SetBorderMode(0);
  c1_2->SetBorderSize(2);
  c1_2->SetLeftMargin(0.16);
  c1_2->SetRightMargin(0.05);
  c1_2->SetTopMargin(0.01);
  c1_2->SetBottomMargin(0.28);
  
  c1_2->SetFrameBorderMode(0);
  c1_2->SetFrameBorderMode(0);
  
  TH1F *Graph_Graph02 = (TH1F *) Graph_Graph01->Clone();
  Graph_Graph02->SetMinimum(0);
  Graph_Graph02->SetMaximum(1.1);
  /// ADD++++++++++++++++++++++++++++++
  //if(cent=="cent8")Graph_Graph02->SetMaximum(3.8);
  if(ss=="2015"){
    Graph_Graph02->SetMaximum(1.8);
  }
  Graph_Graph02->GetXaxis()->SetTitle("b-jet efficiency");
  Graph_Graph02->GetXaxis()->SetLabelFont(42);
  Graph_Graph02->GetXaxis()->SetLabelSize(0.12);
  Graph_Graph02->GetXaxis()->SetTitleSize(0.15);
  Graph_Graph02->GetXaxis()->SetTickSize(0.08);
  Graph_Graph02->GetXaxis()->SetTitleOffset(0.8);
  Graph_Graph02->GetYaxis()->SetTitle("ratio to def");
  Graph_Graph02->GetYaxis()->SetNdivisions(505);
  Graph_Graph02->GetYaxis()->SetLabelSize(0.12);
  Graph_Graph02->GetYaxis()->SetTitleSize(0.14);
  Graph_Graph02->GetYaxis()->SetTitleOffset(0.4);
  Graph_Graph02->Draw();

   TGraph *ratio_roc_b[4];
   //for(int i=0;i<N-1;i++){
   for(int i=0;i<4;i++){
     ratio_roc_b[i]=GetRatioGraph(roc_b[i+2],roc_b[i]);//(roc_b[i],roc_b[0])
     ratio_roc_b[i]->SetFillColor(0);
     ratio_roc_b[i]->SetFillStyle(1000);
     ratio_roc_b[i]->SetMarkerStyle(1);
     ratio_roc_b[i]->SetLineWidth(3);  
     ratio_roc_b[i]->SetMarkerSize(1.);
    
     if(i==0){     
       ratio_roc_b[i]->SetLineColor(8);
       ratio_roc_b[i]->SetLineStyle(10);
       ratio_roc_b[i]->SetMarkerColor(4);
       ratio_roc_b[i]->SetHistogram(Graph_Graph02);
       ratio_roc_b[i]->Draw("samePl");
       ll1->AddEntry(ratio_roc_b[i],"Ratio of Uncaled to Scaled Light Jets");
     }
     else if(i==1) {
       ratio_roc_b[i]->SetLineColor(2);
       ratio_roc_b[i]->SetLineStyle(1);
       ratio_roc_b[i]->SetMarkerColor(2);      
       ratio_roc_b[i]->Draw("samelP");
       ll1->AddEntry(ratio_roc_b[i], "Ratio of Uncaled to Scaled C Jets");
     }
     /* else if(i==2) {
       ratio_roc_b[i]->SetLineColor(kCyan);
       ratio_roc_b[i]->SetLineStyle(2);
       ratio_roc_b[i]->SetMarkerColor(kCyan);      
       ratio_roc_b[i]->Draw("samelP");
     }
     else if(i==3) {
       ratio_roc_b[i]->SetLineColor(3);
       ratio_roc_b[i]->SetLineStyle(1);
       ratio_roc_b[i]->SetMarkerColor(3);      
       ratio_roc_b[i]->Draw("samelP");
     } */
   }

  TLine *line_at_1 = new TLine(Graph_Graph02->GetXaxis()->GetXmin(), 1.0,
                             Graph_Graph02->GetXaxis()->GetXmax(), 1.0);
  line_at_1->SetLineColor(kGray); // Set the line color to gray
  line_at_1->SetLineStyle(1);     // Set the line style to dashed
  line_at_1->Draw("same");
   
  /* TString out;
  out="COMBO/roc_";
  out+="jz_";
  //out+=cent+"_";
  out+=cent+"_";
  out+=ss;
  
  //out+="c.pdf";
  out+=".pdf";
  c1->SaveAs(out); */
  c1->Print("COMBO/plot.pdf");
}

TGraph * GetRatioGraph(TGraph *roc_b1,TGraph *roc_b0){
  int Npt = roc_b0->GetN();
  //TGraph *rat_roc = new TGraph(0);
  TGraph *rat_roc=(TGraph *)roc_b0->Clone();
  double roc,eff;
  for(int jpt=0;jpt < Npt;jpt++){
    roc_b0 -> GetPoint(jpt,eff,roc);
    double roc_eval = roc_b1->Eval(eff);
    double rat = roc_eval/roc;
    //rat_roc->AddPoint(eff,rat);
    //if(!IsNaN(roc))rat_roc->AddPoint(jpt,eff,rat);
    rat_roc->SetPoint(jpt,eff,rat);
  }
  
  for(int jpt=Npt;jpt > 0;jpt--){
    rat_roc -> GetPoint(jpt-1,eff,roc);
    if(IsNaN(roc))rat_roc -> RemovePoint(jpt-1);
  }
  return rat_roc;
}

bool IsNaN(double t)
{
    return t != t;
}
