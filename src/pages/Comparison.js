import React, { useState } from "react";
import useIrisData from "../Components/Iris";
import ComparisonInput from "../Components/ComparisonInput";
import ModelProfile from "../Components/ModelProfile";
import TrainingTime from "../Components/TrainingTime";
import AccuracyComparison from "../Components/AccuracyComparison";
import MAEComparison from "../Components/MAEComparison";
import RMSEComparison from "../Components/RMSEComparison";
import MSEComparison from "../Components/MSEComparison";
import OverfittingGraph from "../Components/OverfittingGraph";
import ComparisonGraph from "../Components/ComparisonGraph";
import ResidualPlot from "../Components/ResidualPlot";
import "../css/Comparison.css";
import ComparisonSummary from "../Components/ComparisonSummary";
import Navbar from "../Components/Navigation";  // Import the Navbar component
function calcMetrics(y, pred) {
  const n = y.length;
  let mse = 0, mae = 0;
  let ssRes = 0, ssTot = 0;
  const meanY = y.reduce((a, b) => a + b, 0) / n;
  for (let i = 0; i < n; i++) {
    const e = y[i] - pred[i];
    mse += e * e;
    mae += Math.abs(e);
    ssRes += e * e;
    ssTot += (y[i] - meanY) ** 2;
  }
  mse /= n;
  mae /= n;
  const rmse = Math.sqrt(mse);
  const r2 = 1 - ssRes / ssTot;
  return { mse, mae, rmse, r2 };
}

function Comparison() {
  const irisData = useIrisData();
  const [lrResult, setLRResult] = useState(null);
  const [rfResult, setRFResult] = useState(null);
  const [lrTime, setLRTime] = useState(null);
  const [rfTime, setRFTime] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleRun = async (cfg) => {
    setLoading(true);

    const lrStart = performance.now();
    const lrRes = await fetch("http://localhost:5000/LR_RF", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data: irisData, xColumn: cfg.x, yColumn: cfg.y, model: 1 }),
    });
    const lrData = await lrRes.json();
    setLRTime(parseFloat((performance.now() - lrStart).toFixed(1)));
    setLRResult(lrData);

    const rfStart = performance.now();
    const rfRes = await fetch("http://localhost:5000/LR_RF", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data: irisData, xColumn: cfg.x, yColumn: cfg.y, trees: cfg.trees, depth: cfg.depth, model: 2 }),
    });
    const rfData = await rfRes.json();
    setRFTime(parseFloat((performance.now() - rfStart).toFixed(1)));
    setRFResult(rfData);

    setLoading(false);
  };

  const lrM = lrResult ? calcMetrics(lrResult.y, lrResult.predictions) : null;
  const rfM = rfResult ? calcMetrics(rfResult.y, rfResult.predictions) : null;

  return (
    <div className="cmp-page">
      <Navbar />  {/* Add the Navbar at the top of the page */}
      <div className="cmp-hero">
        <h1 className="cmp-title">
          <span className="cmp-lr">Linear Regression</span>
          <span className="cmp-vs">VS</span>
          <span className="cmp-rf">Random Forest</span>
        </h1>
        <p className="cmp-sub">Iris Dataset · Regression Performance Comparison</p>
      </div>

      <ComparisonInput onRun={handleRun} />

      {loading && (
        <div className="cmp-loading">
          <div className="cmp-spinner" />
          <p>Running both models...</p>
        </div>
      )}

      {!lrResult && !loading && (
        <div className="cmp-empty">
          <span className="cmp-empty-icon">⚡</span>
          <p>Select columns and click <strong>Run Comparison</strong> to see both models side by side.</p>
        </div>
      )}

      {lrResult && rfResult && lrM && rfM && (
        <div className="cmp-sections">
          <ModelProfile lrM={lrM} rfM={rfM} />
          <TrainingTime lrTime={lrTime} rfTime={rfTime} />
          <AccuracyComparison lrM={lrM} rfM={rfM} />
          <MAEComparison lrM={lrM} rfM={rfM} />
          <RMSEComparison lrM={lrM} rfM={rfM} />
          <MSEComparison lrM={lrM} rfM={rfM} />
          <OverfittingGraph lrM={lrM} rfM={rfM} />
          <ComparisonGraph lr={lrResult} rf={rfResult} />
          <ResidualPlot lr={lrResult} rf={rfResult} />
          <ComparisonSummary lrM={lrM} rfM={rfM} lrTime={lrTime} rfTime={rfTime} />
        </div>
      )}

      
    </div>
  );
}

export default Comparison;