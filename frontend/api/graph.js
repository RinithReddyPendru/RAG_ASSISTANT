import fs from 'fs';
import path from 'path';

export default function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  try {
    const candidates = [
      path.join(process.cwd(), 'frontend', 'graph_data.json'),
      path.join(process.cwd(), 'graph_data.json')
    ];
    let data = null;
    for (const p of candidates) {
      if (fs.existsSync(p)) {
        data = JSON.parse(fs.readFileSync(p, 'utf8'));
        break;
      }
    }

    if (!data) {
      return res.status(404).json({ error: 'Graph data not found' });
    }

    return res.status(200).json(data);
  } catch (err) {
    return res.status(500).json({ error: err.message });
  }
}
