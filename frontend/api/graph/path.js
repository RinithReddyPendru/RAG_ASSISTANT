import fs from 'fs';
import path from 'path';

export default function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  const { source, target } = req.query;
  if (!source || !target) {
    return res.status(400).json({ found: false, detail: 'source and target parameters are required' });
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
      return res.status(500).json({ found: false, detail: 'Graph data file not found' });
    }

    const s = source.trim().toLowerCase();
    const t = target.trim().toLowerCase();

    const nodeLookup = {};
    data.nodes.forEach(n => {
      nodeLookup[n.id.toLowerCase()] = n.id;
    });

    const realSource = nodeLookup[s];
    const realTarget = nodeLookup[t];

    if (!realSource || !realTarget) {
      return res.status(404).json({
        found: false,
        detail: `Entity "${!realSource ? source : target}" was not found in the graph.`
      });
    }

    const adj = {};
    data.nodes.forEach(n => adj[n.id] = []);
    data.edges.forEach(e => {
      if (!adj[e.from]) adj[e.from] = [];
      if (!adj[e.to]) adj[e.to] = [];
      adj[e.from].push({ to: e.to, label: e.label });
      adj[e.to].push({ to: e.from, label: e.label });
    });

    const queue = [[realSource]];
    const visited = new Set([realSource]);

    while (queue.length > 0) {
      const currentPath = queue.shift();
      const curr = currentPath[currentPath.length - 1];

      if (curr === realTarget) {
        const pathEdges = [];
        const narrativeParts = [];
        for (let i = 0; i < currentPath.length - 1; i++) {
          const u = currentPath[i];
          const v = currentPath[i + 1];
          const edgeObj = (adj[u] || []).find(e => e.to === v);
          const rel = edgeObj ? edgeObj.label : 'related_to';
          pathEdges.push({ from: u, to: v, label: rel });
          narrativeParts.push(`[${u}] --(${rel})--> [${v}]`);
        }
        return res.status(200).json({
          found: true,
          hops: currentPath.length - 1,
          path_nodes: currentPath,
          path_edges: pathEdges,
          narrative: narrativeParts.join(' ➔ ')
        });
      }

      for (const neighbor of (adj[curr] || [])) {
        if (!visited.has(neighbor.to)) {
          visited.add(neighbor.to);
          queue.push([...currentPath, neighbor.to]);
        }
      }
    }

    return res.status(200).json({
      found: false,
      detail: `No path found between ${realSource} and ${realTarget}.`
    });
  } catch (err) {
    return res.status(500).json({ found: false, detail: err.message });
  }
}
