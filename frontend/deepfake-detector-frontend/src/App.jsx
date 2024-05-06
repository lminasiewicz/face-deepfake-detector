import {useState} from 'react';
import './App.scss';
import NavBar from "./NavBar";
import axios from 'axios';

function App() {
  const [file, setFile] = useState();
  const [response, setResponse] = useState();
  const [error, setError] = useState();

  function handleChange(event) {
    setFile(event.target.files[0])
  }
  
  function handleSubmit(event) {
    event.preventDefault();
    setError(null);
    setResponse(null);
    console.log(file.name.endsWith(".jpg"))
    if (file.name.endsWith(".jpg") || file.name.endsWith(".png") || file.name.endsWith(".bmp")) {
      const url = 'http://localhost:5000/is_deepfake';
      const formData = new FormData();
      formData.append('file', file);
      formData.append('fileName', file.name);
      const config = {
        headers: {
          'content-type': 'multipart/form-data',
        },
      };
      axios.post(url, formData, config).then((response) => {
        console.log(response.data);
        setResponse(response.data);
      })
      .catch((err) => {
        console.error("Error uploading file: ", err.message);
        setError(err);
      });
    }
    else {
      setError("File must be an image file! (.jpg, .png, .bmp)")
    }
  }


  return (
    <>
      <NavBar/>
      <div className="App">
        <div className="main">
          <h className="util-element util-header">
            Face Deepfake Detector
          </h>
          <div className="util-element util-body">
            <p>Seen an image of a human face somewhere on the internet, and want to know if it's real? This app employs a 260000+ parameter AI model that detects deepfakes among images of human faces. Simply upload your file below (.jpg, .png, or .bmp) and get a simple verdict. Don't get scammed with accounts of fake people who don't exist! Try it out for free below:</p>
            <form className="util-form" onSubmit={handleSubmit}>
              <input type="file" onChange={handleChange}/><br/>
              <button type="submit">Upload</button>
            </form>
            <div className="util-reports">
              {error && <p>Error uploading file: {error}</p>}
              {response && <p>Verdict: {response.verdict}</p>}
              {response && <p>Certainty: around {response.certainty}%</p>}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

export default App;
