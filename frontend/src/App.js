import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [dragActive, setDragActive] = useState(false);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setResult(null);
      
      // Create preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(selectedFile);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      setFile(droppedFile);
      setResult(null);
      
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(droppedFile);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      alert("Please select a file first!");
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post("http://localhost:5050/api/infer", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(res.data);
    } catch (err) {
      console.error(err);
      alert("Error uploading or analyzing file");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
  };

  const getFakeStatus = () => {
    if (!result) return null;
    const probability = result.fake_probability * 100;
    
    if (probability < 30) {
      return { status: "AUTHENTIC", color: "#10b981", icon: "✓" };
    } else if (probability < 70) {
      return { status: "UNCERTAIN", color: "#f59e0b", icon: "⚠" };
    } else {
      return { status: "FAKE DETECTED", color: "#ef4444", icon: "✕" };
    }
  };

  const statusInfo = getFakeStatus();

  return (
    <div className="app">
      <div className="background-grid"></div>
      
      <header className="header">
        <div className="logo">
          <div className="logo-icon">🎭</div>
          <h1>DeepGuard</h1>
        </div>
        <p className="tagline">AI-Powered Deepfake Detection</p>
      </header>

      <main className="main-content">
        {!preview ? (
          <div 
            className={`upload-zone ${dragActive ? 'drag-active' : ''}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <div className="upload-icon">📁</div>
            <h2>Drop your image here</h2>
            <p>or click to browse</p>
            <input 
              type="file" 
              accept="image/*" 
              onChange={handleFileChange}
              className="file-input"
              id="file-upload"
            />
            <label htmlFor="file-upload" className="upload-button">
              Choose File
            </label>
            <div className="supported-formats">
              Supported: JPG, PNG, WEBP
            </div>
          </div>
        ) : (
          <div className="analysis-container">
            <div className="preview-section">
              <img src={preview} alt="Preview" className="preview-image" />
              <button onClick={handleReset} className="reset-button">
                ← Upload Different Image
              </button>
            </div>

            <div className="results-section">
              {!result ? (
                <div className="analyze-prompt">
                  <h2>Ready to Analyze</h2>
                  <p>Click the button below to detect if this image is authentic or AI-generated</p>
                  <button
                    onClick={handleUpload}
                    disabled={loading}
                    className="analyze-button"
                  >
                    {loading ? (
                      <>
                        <span className="spinner"></span>
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <span>🔍</span>
                        Analyze Image
                      </>
                    )}
                  </button>
                </div>
              ) : (
                <div className="results">
                  <div className={`status-badge ${statusInfo.status.toLowerCase().replace(' ', '-')}`}>
                    <span className="status-icon">{statusInfo.icon}</span>
                    <span className="status-text">{statusInfo.status}</span>
                  </div>

                  <div className="probability-display">
                    <div className="probability-label">Deepfake Probability</div>
                    <div className="probability-value" style={{ color: statusInfo.color }}>
                      {(result.fake_probability * 100).toFixed(1)}%
                    </div>
                    <div className="probability-bar">
                      <div 
                        className="probability-fill" 
                        style={{ 
                          width: `${result.fake_probability * 100}%`,
                          backgroundColor: statusInfo.color
                        }}
                      ></div>
                    </div>
                  </div>

                  <div className="metrics-grid">
                    <div className="metric-card">
                      <div className="metric-label">Model Used</div>
                      <div className="metric-value">{result.model || 'ResNet-18'}</div>
                    </div>
                    <div className="metric-card">
                      <div className="metric-label">Confidence</div>
                      <div className="metric-value">
                        {(Math.max(result.fake_probability, 1 - result.fake_probability) * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>

                  <div className="interpretation">
                    <h3>What does this mean?</h3>
                    <p>
                      {result.fake_probability < 0.3 
                        ? "This image appears to be authentic with high confidence. No significant signs of AI manipulation detected."
                        : result.fake_probability < 0.7
                        ? "The analysis is inconclusive. The image shows some characteristics that could indicate AI generation, but it's not definitive."
                        : "This image shows strong indicators of being AI-generated or manipulated. Multiple deepfake signatures detected."}
                    </p>
                  </div>

                  <button onClick={handleReset} className="analyze-another">
                    Analyze Another Image
                  </button>
                </div>
              )}
            </div>
          </div>
        )}
      </main>

      <footer className="footer">
        <p>Powered by Advanced Neural Networks • Built with React & PyTorch</p>
      </footer>
    </div>
  );
}

export default App;