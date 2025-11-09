import { useEffect, useState } from "react";
import SearchBar from "./components/SearchBar"
import { Prediction } from "./components/Prediction";
import { Confidence } from "./components/Confidence";

function App() {
  const [prediction,setPrediction] = useState(0);
  const [probability,setProbability] = useState(0);
  
  const [email,setEmail] = useState("");
  useEffect(()=>{
    if(!email){return};
    fetch("http://localhost:8000/predict_phishing", {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json',
      },
      body: JSON.stringify({"text" : email})
    })
    .then(response =>{
      if (!response.ok){
        throw new Error("something wrong with response")
      }
      return response.json();
    })
    .then((vals) =>{
      console.log(vals);
      if(vals != undefined){
        setPrediction(vals["prediction"])
        setProbability(vals["confidence"]);
      }
    })
    .catch((e) =>{
      console.log("ERROR: " + e);
    })
  },[email])

  return (
    <>
      <>
        <h1>
          Phishing Scam AI Detector
        </h1>
        <SearchBar email={email} setEmail={setEmail}></SearchBar>
        <Prediction prediction={prediction}></Prediction>
        <Confidence confidence={probability}></Confidence>
      </>
    </>
  )
}

export default App
