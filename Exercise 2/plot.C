void plot(){

  // opening files
  TFile f1("out_dat.root");
  TFile f2("out_sig.root");
  TFile f3("out_bkg.root");

  TLegend *leg = new TLegend(0.7, 0.1, 0.9, 0.3);

  Double_t factor = 1.;

  // get histograms from the files
  TH1F * h_n_1 = (TH1F*)f1.Get("h_nparticles");
  TH1F * h_n_2 = (TH1F*)f2.Get("h_nparticles");
  TH1F * h_n_3 = (TH1F*)f3.Get("h_nparticles");

  //scaling
  h_n_1->Scale(factor/h_n_1->Integral());
  h_n_2->Scale(factor/h_n_2->Integral());
  h_n_3->Scale(factor/h_n_3->Integral());
  
  // set line colors
  h_n_1->SetLineColor(1);
  h_n_2->SetLineColor(2);
  h_n_3->SetLineColor(3);
  
  TH1F * h_x_1 = (TH1F*)f1.Get("h_x");
  TH1F * h_x_2 = (TH1F*)f2.Get("h_x");
  TH1F * h_x_3 = (TH1F*)f3.Get("h_x");
  h_x_1->Scale(factor/h_x_1->Integral());
  h_x_2->Scale(factor/h_x_2->Integral());
  h_x_3->Scale(factor/h_x_3->Integral());
  h_x_1->SetLineColor(1);
  h_x_2->SetLineColor(2);
  h_x_3->SetLineColor(3);

  TH1F * h_r_1 = (TH1F*)f1.Get("h_r");
  TH1F * h_r_2 = (TH1F*)f2.Get("h_r");
  TH1F * h_r_3 = (TH1F*)f3.Get("h_r");
  h_r_1->Scale(factor/h_r_1->Integral());
  h_r_2->Scale(factor/h_r_2->Integral());
  h_r_3->Scale(factor/h_r_3->Integral());
  h_r_1->SetLineColor(1);
  h_r_2->SetLineColor(2);
  h_r_3->SetLineColor(3);

  TH1F * h_mulike_1 = (TH1F*)f1.Get("h_mulike");
  TH1F * h_mulike_2 = (TH1F*)f2.Get("h_mulike");
  TH1F * h_mulike_3 = (TH1F*)f3.Get("h_mulike");
  h_mulike_1->Scale(factor/h_mulike_1->Integral());
  h_mulike_2->Scale(factor/h_mulike_2->Integral());
  h_mulike_3->Scale(factor/h_mulike_3->Integral());
  h_mulike_1->SetLineColor(1);
  h_mulike_2->SetLineColor(2);
  h_mulike_3->SetLineColor(3);
  //h_mulike_1->GetYaxis()->SetRangeUser(0.,300);

  TH1F * h_mass_1 = (TH1F*)f1.Get("h_mass");
  TH1F * h_mass_2 = (TH1F*)f2.Get("h_mass");
  TH1F * h_mass_3 = (TH1F*)f3.Get("h_mass");
  h_mass_1->Scale(factor/h_mass_1->Integral());
  h_mass_2->Scale(factor/h_mass_2->Integral());
  h_mass_3->Scale(factor/h_mass_3->Integral());
  h_mass_1->SetLineColor(1);
  h_mass_2->SetLineColor(2);
  h_mass_3->SetLineColor(3);
  leg->AddEntry(h_mass_1, "Data", "p");
  leg->AddEntry(h_mass_2, "Signal", "l");
  leg->AddEntry(h_mass_3, "Background", "l");
  TF1 *fit = new TF1("fit", "gausn", 15, 25);//19,20
  h_mass_1->Fit(fit, "R");

  TH1F * h_theta_1 = (TH1F*)f1.Get("h_theta");
  TH1F * h_theta_2 = (TH1F*)f2.Get("h_theta");
  TH1F * h_theta_3 = (TH1F*)f3.Get("h_theta");
  h_theta_1->Scale(factor/h_theta_1->Integral());
  h_theta_2->Scale(factor/h_theta_2->Integral());
  h_theta_3->Scale(factor/h_theta_3->Integral());
  h_theta_1->SetLineColor(1);
  h_theta_2->SetLineColor(2);
  h_theta_3->SetLineColor(3);

  TH1F * h_eta_1 = (TH1F*)f1.Get("h_eta");
  TH1F * h_eta_2 = (TH1F*)f2.Get("h_eta");
  TH1F * h_eta_3 = (TH1F*)f3.Get("h_eta");
  h_eta_1->Scale(factor/h_eta_1->Integral());
  h_eta_2->Scale(factor/h_eta_2->Integral());
  h_eta_3->Scale(factor/h_eta_3->Integral());
  h_eta_1->SetLineColor(1);
  h_eta_2->SetLineColor(2);
  h_eta_3->SetLineColor(3);

  TH1F * h_phi_1 = (TH1F*)f1.Get("h_phi");
  TH1F * h_phi_2 = (TH1F*)f2.Get("h_phi");
  TH1F * h_phi_3 = (TH1F*)f3.Get("h_phi");
  h_phi_1->Scale(factor/h_phi_1->Integral());
  h_phi_2->Scale(factor/h_phi_2->Integral());
  h_phi_3->Scale(factor/h_phi_3->Integral());
  h_phi_1->SetLineColor(1);
  h_phi_2->SetLineColor(2);
  h_phi_3->SetLineColor(3);

  TH1F * h_pt_1 = (TH1F*)f1.Get("h_pt");
  TH1F * h_pt_2 = (TH1F*)f2.Get("h_pt");
  TH1F * h_pt_3 = (TH1F*)f3.Get("h_pt");
  h_pt_1->Scale(factor/h_pt_1->Integral());
  h_pt_2->Scale(factor/h_pt_2->Integral());
  h_pt_3->Scale(factor/h_pt_3->Integral());
  h_pt_1->SetLineColor(1);
  h_pt_2->SetLineColor(2);
  h_pt_3->SetLineColor(3);

  TH1F * h_leadingpt_1 = (TH1F*)f1.Get("h_leadingpt");
  TH1F * h_leadingpt_2 = (TH1F*)f2.Get("h_leadingpt");
  TH1F * h_leadingpt_3 = (TH1F*)f3.Get("h_leadingpt");
  h_leadingpt_1->Scale(factor/h_leadingpt_1->Integral());
  h_leadingpt_2->Scale(factor/h_leadingpt_2->Integral());
  h_leadingpt_3->Scale(factor/h_leadingpt_3->Integral());
  h_leadingpt_1->SetLineColor(1);
  h_leadingpt_2->SetLineColor(2);
  h_leadingpt_3->SetLineColor(3);

  TH1F * h_subleadingpt_1 = (TH1F*)f1.Get("h_subleadingpt");
  TH1F * h_subleadingpt_2 = (TH1F*)f2.Get("h_subleadingpt");
  TH1F * h_subleadingpt_3 = (TH1F*)f3.Get("h_subleadingpt");
  h_subleadingpt_1->Scale(factor/h_subleadingpt_1->Integral());
  h_subleadingpt_2->Scale(factor/h_subleadingpt_2->Integral());
  h_subleadingpt_3->Scale(factor/h_subleadingpt_3->Integral());
  h_subleadingpt_1->SetLineColor(1);
  h_subleadingpt_2->SetLineColor(2);
  h_subleadingpt_3->SetLineColor(3);

  /* TH1F * h_corrpt_1 = (TH1F*)f1.Get("h_leadingpt:h_subleadingpt");
  TH1F * h_corrpt_2 = (TH1F*)f2.Get("h_leadingpt:h_subleadingpt");
  TH1F * h_corrpt_3 = (TH1F*)f3.Get("h_leadingpt:h_subleadingpt");
  h_corrpt_1->Scale(factor/h_corrpt_1->Integral());
  h_corrpt_2->Scale(factor/h_corrpt_2->Integral());
  h_corrpt_3->Scale(factor/h_corrpt_3->Integral()); 
  h_corrpt_1->SetLineColor(1);
  h_corrpt_2->SetLineColor(2);
  h_corrpt_3->SetLineColor(3); */
  
  TH1F * h_x_y_1 = (TH1F*)f1.Get("h_x_y");
  TH1F * h_x_y_2 = (TH1F*)f2.Get("h_x_y");
  TH1F * h_x_y_3 = (TH1F*)f3.Get("h_x_y");

  TH1F * h_r1_r2_1 = (TH1F*)f1.Get("h_r1_r2");
  TH1F * h_r1_r2_2 = (TH1F*)f2.Get("h_r1_r2");
  TH1F * h_r1_r2_3 = (TH1F*)f3.Get("h_r1_r2");

  gStyle->SetOptFit(1111);

  TCanvas c; 
  c.SaveAs("plots.pdf["); // opening pdf
  //c.SetLogy();
  //c.SetLogx();

  // draw all histograms
  h_n_1->Draw("e");
  h_n_2->Draw("same");
  h_n_3->Draw("same");

  c.SaveAs("plots.pdf"); // plot

  // next plot
  h_x_1->Draw("e");
  h_x_2->Draw("same");
  h_x_3->Draw("same");
  c.SaveAs("plots.pdf");

  // next plot
  h_r_1->Draw("e");
  h_r_2->Draw("same");
  h_r_3->Draw("same");
  c.SaveAs("plots.pdf");

  // next plot
  h_mulike_1->Draw("e");
  h_mulike_2->Draw("same");
  h_mulike_3->Draw("same");
  c.SaveAs("plots.pdf");

  // next plot
  h_mass_1->Draw("e");
  h_mass_2->Draw("same");
  h_mass_3->Draw("same");
  leg->Draw();
  c.SaveAs("plots.pdf");

  // next plot
  h_theta_1->Draw("e");
  h_theta_2->Draw("same");
  h_theta_3->Draw("same");
  c.SaveAs("plots.pdf");

  // next plot
  h_eta_1->Draw("e");
  h_eta_2->Draw("same");
  h_eta_3->Draw("same");
  c.SaveAs("plots.pdf");

  // next plot
  h_phi_1->Draw("e");
  h_phi_2->Draw("same");
  h_phi_3->Draw("same");
  c.SaveAs("plots.pdf");

  // next plot
  h_pt_1->Draw("e");
  h_pt_2->Draw("same");
  h_pt_3->Draw("same");
  c.SaveAs("plots.pdf");

  // next plot
  h_leadingpt_1->Draw("e");
  h_leadingpt_2->Draw("same");
  h_leadingpt_3->Draw("same");
  c.SaveAs("plots.pdf");

  // next plot
  h_subleadingpt_1->Draw("e");
  h_subleadingpt_2->Draw("same");
  h_subleadingpt_3->Draw("same");
  c.SaveAs("plots.pdf");

  /* // next plot
  h_corrpt_1->Draw("e");
  h_corrpt_2->Draw("same");
  h_corrpt_3->Draw("same");
  c.SaveAs("plots.pdf"); */

  // next plot
  c.Clear();
  c.Divide(3);
  c.cd(1); h_x_y_1->Draw("colz");
  c.cd(2); h_x_y_2->Draw("colz");
  c.cd(3); h_x_y_3->Draw("colz");
  c.SaveAs("plots.pdf");

  // next plot
  c.Clear();
  c.Divide(3);
  c.cd(1); h_r1_r2_1->Draw("colz");
  c.cd(2); h_r1_r2_2->Draw("colz");
  c.cd(3); h_r1_r2_3->Draw("colz");
  c.SaveAs("plots.pdf");

  c.SaveAs("plots.pdf]"); // closing pdf
}
