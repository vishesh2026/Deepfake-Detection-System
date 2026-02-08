const express = require('express');
const cors = require('cors');
const multer = require('multer');
const { execFileSync } = require('child_process');
const path = require('path');
const fs = require('fs');
const jwt = require('jsonwebtoken');
const app = express();
app.use(express.json()); app.use(cors());
const upload = multer({ dest: 'uploads/' });
const JWT_SECRET = process.env.JWT_SECRET || 'devsecret';
app.post('/auth/signup', (req,res)=>{ return res.json({token:'demo-token'}); });
app.post('/auth/login', (req,res)=>{ return res.json({token:'demo-token'}); });
app.post('/api/infer', upload.single('file'), async (req, res) => {
  try {
    const input = req.file.path;
    const py =
  process.platform === 'win32'
    ? path.resolve(__dirname, '..', 'venv', 'Scripts', 'python.exe')
    : path.resolve(__dirname, '..', 'venv', 'bin', 'python');

    const script = path.resolve(__dirname, '..', 'ml', 'inference', 'infer.py');
    const modelPath = path.resolve(__dirname, '..', 'models', 'resnet18_finetuned.pth');

    const out = execFileSync(py, [script, '--image', input, '--model', modelPath], {
      encoding: 'utf8',
      maxBuffer: 50 * 1024 * 1024
    });

    fs.unlinkSync(input); // Clean up uploaded file

    res.json(JSON.parse(out));
  } catch (e) {
    console.error('Error during inference:', e);
    res.status(500).json({ error: String(e) });
  }
});

app.get('/health',(req,res)=>res.json({status:'ok'}));
app.use(express.static(path.join(__dirname,'..','frontend','public')));
app.listen(5050, ()=>console.log('Server running on 5050'));
