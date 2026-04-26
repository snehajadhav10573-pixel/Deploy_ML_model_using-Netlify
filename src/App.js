import { BrowserRouter, Routes, Route } from "react-router-dom";

import Home from "./pages/Home";
import LinearReg from "./pages/LinearReg";
import RandomForest from "./pages/RandomForest";
import Comparison from "./pages/Comparison";
import Classification from "./pages/Classification";  // 👈 add this

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/LinearRegression" element={<LinearReg />} />
        <Route path="/RandomForest" element={<RandomForest />} />
        <Route path="/Comparison" element={<Comparison />} />
        <Route path="/Classification" element={<Classification />} />  {/* 👈 add this */}
      </Routes>
    </BrowserRouter>
  );
}

export default App;