import { useState } from "react";
import SearchBar from "./components/SearchBar"

function App() {
  const [email,setEmail] = useState("");
  return (
    <>
      <h1>
        Phishing Scam AI Detector
      </h1>
      <SearchBar email={email} setEmail={setEmail}></SearchBar>
    </>
  )
}

export default App
