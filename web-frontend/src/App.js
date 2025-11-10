
import React, { useState, useRef, useEffect } from 'react';
import {
  Container, Typography, Box, Button, CircularProgress, Card, CardContent, Avatar, Fade, Alert
} from '@mui/material';
import PestControlIcon from '@mui/icons-material/BugReport';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [useCamera, setUseCamera] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [liveResult, setLiveResult] = useState(null);
  const DETECTION_THRESHOLD = 0.5;
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const pollRef = useRef(null);
  const noPestCountRef = useRef(0);

  const handleFileChange = (e) => {
    const f = e.target.files[0];
    setFile(f);
    setResult(null);
    setError(null);
    if (f) {
      const reader = new FileReader();
      reader.onload = () => setPreview(reader.result);
      reader.readAsDataURL(f);
    } else {
      setPreview(null);
    }
  };

  // start camera stream
  const startCamera = async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
      // First set cameraActive so the <video> element is rendered and ref attached
      setCameraActive(true);
      // wait until next paint so the video ref is available
      await new Promise((resolve) => requestAnimationFrame(() => resolve()));
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        // some browsers require play() to be called after srcObject is set
        try { await videoRef.current.play(); } catch (e) { /* ignore play() exceptions */ }
        // start polling
        startPolling();
      }
    } catch (err) {
      setError('Unable to access camera: ' + err.message);
    }
  };

  const stopCamera = () => {
    try {
      if (videoRef.current && videoRef.current.srcObject) {
        const tracks = videoRef.current.srcObject.getTracks();
        tracks.forEach(t => t.stop());
        videoRef.current.srcObject = null;
      }
    } catch (e) {
      // ignore
    }
    setCameraActive(false);
    stopPolling();
    setLiveResult(null);
  };

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  const startPolling = () => {
    // poll every 1000ms
    if (pollRef.current) return;
    pollRef.current = setInterval(captureAndPredict, 1000);
  };

  const stopPolling = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };

  const captureAndPredict = async () => {
    if (!videoRef.current) return;
    const video = videoRef.current;
    const w = 224, h = 224;
    // use an offscreen canvas for capture so we don't overwrite the visible preview canvas
    let offscreen = canvasRef.current && canvasRef.current.offscreen ? canvasRef.current.offscreen : null;
    if (!offscreen) {
      offscreen = document.createElement('canvas');
      // keep reference on the visible canvasRef.current so we can still show preview
      if (!canvasRef.current) canvasRef.current = null;
      // store under a property to avoid clobbering the react ref
      if (canvasRef.current) canvasRef.current.offscreen = offscreen;
      else canvasRef.current = { offscreen };
    }
    const canvas = offscreen;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    // draw centered crop
    ctx.drawImage(video, 0, 0, w, h);
    canvas.toBlob(async (blob) => {
      if (!blob) return;
      const fd = new FormData();
      fd.append('file', blob, 'frame.jpg');
      try {
        const res = await fetch('http://localhost:8000/predict', { method: 'POST', body: fd });
        const data = await res.json();
        setLiveResult(data);
        // decide pest on/off based on class/confidence
        const confidence = Number(data.confidence || 0);
        // consider pest present if confidence > 0.5 by default
        if (confidence > 0.5) {
          noPestCountRef.current = 0;
          // notify backend explicitly to ensure buzzer stays on (duration param optional)
          await fetch('http://localhost:8000/pest', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ pest: true, duration: Math.max(1, Math.round(confidence * 10)) }) });
        } else {
          // increment no-pest counter; require 2 consecutive misses to turn off to avoid flicker
          noPestCountRef.current += 1;
          if (noPestCountRef.current >= 2) {
            await fetch('http://localhost:8000/pest', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ pest: false }) });
          }
        }
      } catch (err) {
        console.error('Live predict failed', err);
      }
    }, 'image/jpeg', 0.8);
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError('Prediction failed. Backend not reachable.');
    }
    setLoading(false);
  };

  return (
    <Container maxWidth="sm" sx={{ mt: 8 }}>
      <Box sx={{ minHeight: '28vh', display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 2 }}>
        <Box sx={{ textAlign: 'center' }}>
          <Avatar sx={{ bgcolor: 'transparent', width: 88, height: 88, mx: 'auto', mb: 2, boxShadow: 3 }}>
            <PestControlIcon fontSize="large" sx={{ color: '#0f172a', fontSize: 36 }} />
          </Avatar>
          <Typography variant="h3" fontWeight={800} sx={{ letterSpacing: 1, color: '#0f172a' }} gutterBottom>
            Pest Detection
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Live feed monitor — showing only actionable state
          </Typography>
        </Box>
      </Box>
      <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 3 }}>
        <Box sx={{ display: 'flex', gap: 2, width: '100%' }}>
          <Button variant="outlined" component="label" sx={{ flex: 1 }}>
            Select Image
            <input type="file" accept="image/*" hidden onChange={handleFileChange} />
          </Button>
          <Button variant={cameraActive ? 'contained' : 'outlined'} color={cameraActive ? 'secondary' : 'primary'} onClick={() => { if (!cameraActive) startCamera(); else stopCamera(); }} sx={{ width: 160, textTransform: 'none' }}>
            {cameraActive ? 'Stop Camera' : 'Use Camera'}
          </Button>
        </Box>
        {preview && (
          <Card sx={{ width: '100%', boxShadow: 3 }}>
            <CardContent sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <img src={preview} alt="preview" style={{ maxWidth: '100%', maxHeight: 240, borderRadius: 8 }} />
            </CardContent>
          </Card>
        )}
        <Button variant="contained" color="primary" onClick={handleUpload} disabled={!file || loading} sx={{ width: '100%', borderRadius: 2, textTransform: 'none' }}>
          {loading ? <CircularProgress size={24} /> : 'Run Prediction'}
        </Button>
        {/* Live camera preview when active */}
        {cameraActive && (
          <Card sx={{ width: '100%', boxShadow: 6, mt: 2, borderRadius: 3, overflow: 'hidden' }}>
            <CardContent sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, alignItems: 'center', gap: 2 }}>
              <Box sx={{ flex: 1, width: '100%' }}>
                <video ref={videoRef} style={{ width: '100%', maxHeight: 420, borderRadius: 8, background: '#000' }} playsInline muted autoPlay />
              </Box>
              <Box sx={{ width: { xs: '100%', md: 220 }, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 1 }}>
                {/* Big circular status indicator */}
                <Box sx={{ width: 160, height: 160, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', bgcolor: liveResult && liveResult.confidence > DETECTION_THRESHOLD ? 'error.main' : 'success.main', color: '#fff', boxShadow: 8, transition: 'all 300ms ease' }}>
                  <Typography variant="h5" sx={{ fontWeight: 800, letterSpacing: 1.5 }}>
                    {liveResult && liveResult.confidence > DETECTION_THRESHOLD ? 'PEST' : 'CLEAR'}
                  </Typography>
                </Box>
                <Typography variant="caption" color="text.secondary">{liveResult ? `${Math.round(liveResult.confidence * 100)}% confidence` : 'Waiting for first frame...'}</Typography>
                <Box sx={{ mt: 1 }}>
                  <canvas ref={canvasRef} style={{ width: 120, height: 120, borderRadius: 8, border: '1px solid rgba(15,23,42,0.06)' }} />
                </Box>
              </Box>
            </CardContent>
          </Card>
        )}
        {error && <Alert severity="error" sx={{ width: '100%' }}>{error}</Alert>}
        {result && (
          <Fade in timeout={600}>
            <Card sx={{ mt: 2, width: '100%', boxShadow: 4, borderRadius: 2 }}>
              <CardContent sx={{ textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 1 }}>
                <Typography variant="h6" sx={{ fontWeight: 800 }}>
                  {result.confidence > DETECTION_THRESHOLD ? 'PEST DETECTED' : 'CLEAR'}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {Math.round(result.confidence * 100)}% confidence
                </Typography>
              </CardContent>
            </Card>
          </Fade>
        )}
      </Box>
    </Container>
  );
}

export default App;
