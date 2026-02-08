const express = require('express');
const cors = require('cors');
const multer = require('multer');
const { execFileSync } = require('child_process');
const path = require('path');
const fs = require('fs');
const jwt = require('jsonwebtoken');

const app = express();
app.use(express.json());
app.use(cors({
  origin: ['http://localhost:3000', 'https://deepfake-detection-system-kklf.vercel.app/'], // Add your frontend URLs
  methods: ['GET', 'POST']
}));

const upload = multer({ dest: path.join(__dirname, 'uploads') });
const JWT_SECRET = process.env.JWT_SECRET || 'devsecret';

app.post('/auth/signup', (req,res) => res.json({ token:'demo-token' }));
app.post('/auth/login', (req,res) => res.json({ token:'demo-token' }));

app.post('/api/infer', upload.single('file'), async (req, res) => {
  try {
    const input = req.file.path;

    // Pick correct Python path depending on platform
    const py = process.platform === 'win32'
      ? path.resolve(__dirname, '..', 'venv', 'Scripts', 'python.exe') // Windows
      : 'python3'; // Linux / Render

    const script = path.resolve(__dirname, '..', 'ml', 'inference', 'infer.py');
    const modelPath = path.resolve(__dirname, '..', 'models', 'resnet18_finetuned.pth');

    const out = execFileSync(py, [script, '--image', input, '--model', modelPath], {
      encoding: 'utf8',
      maxBuffer: 50 * 1024 * 1024
    });

    fs.unlinkSync(input); // remove uploaded file

    res.json(JSON.parse(out));
  } catch (e) {
    console.error('Error during inference:', e);
    res.status(500).json({ error: String(e) });
  }
});

app.get('/health', (req,res) => res.json({ status:'ok' }));

// Optional: serve frontend static files
app.use(express.static(path.join(__dirname,'..','frontend','public')));

const PORT = process.env.PORT || 5050;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
