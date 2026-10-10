"""
RoomShift AI — Web Application

Flask-based interface for uploading videos and viewing 3D change detection results.
"""
import os
import json
import logging
import tempfile
import time
from pathlib import Path

from flask import (
    Flask, render_template_string, request, jsonify,
    send_from_directory, redirect, url_for,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB
app.config['UPLOAD_FOLDER'] = os.path.join(tempfile.gettempdir(), 'roomshift_uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(os.getcwd(), 'outputs')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# In-memory job state
_jobs = {}

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RoomShift AI — 3D Spatial Change Detection</title>
    <meta name="description" content="Detect spatial changes between two video recordings of the same indoor environment using 3D reconstruction and computer vision.">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0a1a;
            --bg-secondary: #12122a;
            --bg-card: #1a1a3e;
            --bg-glass: rgba(26, 26, 62, 0.7);
            --accent-1: #6c5ce7;
            --accent-2: #a29bfe;
            --accent-3: #00cec9;
            --text-primary: #e8e8f0;
            --text-secondary: #8888aa;
            --text-muted: #5a5a7a;
            --moved-color: #FF6B35;
            --added-color: #2ECC71;
            --removed-color: #E74C3C;
            --uncertain-color: #95A5A6;
            --geometric-color: #3498DB;
            --border: rgba(108, 92, 231, 0.2);
            --shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            --radius: 16px;
            --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Inter', -apple-system, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
        }
        
        /* Animated background */
        body::before {
            content: '';
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: 
                radial-gradient(ellipse at 20% 50%, rgba(108, 92, 231, 0.08) 0%, transparent 60%),
                radial-gradient(ellipse at 80% 20%, rgba(0, 206, 201, 0.06) 0%, transparent 60%),
                radial-gradient(ellipse at 50% 80%, rgba(162, 155, 254, 0.05) 0%, transparent 60%);
            z-index: 0;
            pointer-events: none;
        }
        
        .container {
            position: relative;
            z-index: 1;
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        /* Header */
        header {
            text-align: center;
            padding: 3rem 0 2rem;
        }
        
        .logo {
            font-size: 2.8rem;
            font-weight: 800;
            background: linear-gradient(135deg, var(--accent-1), var(--accent-3));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.02em;
            margin-bottom: 0.5rem;
        }
        
        .subtitle {
            color: var(--text-secondary);
            font-size: 1.1rem;
            font-weight: 300;
            max-width: 600px;
            margin: 0 auto;
            line-height: 1.6;
        }
        
        /* Cards */
        .card {
            background: var(--bg-glass);
            backdrop-filter: blur(20px);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 2rem;
            margin-bottom: 1.5rem;
            box-shadow: var(--shadow);
            transition: var(--transition);
        }
        
        .card:hover {
            border-color: rgba(108, 92, 231, 0.4);
            transform: translateY(-2px);
        }
        
        .card h2 {
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 1.2rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .card h2 .icon {
            font-size: 1.5rem;
        }
        
        /* Upload area */
        .upload-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }
        
        .upload-zone {
            border: 2px dashed var(--border);
            border-radius: var(--radius);
            padding: 2.5rem 1.5rem;
            text-align: center;
            cursor: pointer;
            transition: var(--transition);
            position: relative;
            overflow: hidden;
        }
        
        .upload-zone:hover, .upload-zone.dragover {
            border-color: var(--accent-1);
            background: rgba(108, 92, 231, 0.08);
        }
        
        .upload-zone.has-file {
            border-color: var(--accent-3);
            border-style: solid;
            background: rgba(0, 206, 201, 0.05);
        }
        
        .upload-zone input {
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            opacity: 0;
            cursor: pointer;
        }
        
        .upload-zone .upload-icon {
            font-size: 2.5rem;
            margin-bottom: 0.8rem;
        }
        
        .upload-zone .upload-label {
            font-weight: 600;
            font-size: 1.1rem;
            margin-bottom: 0.3rem;
        }
        
        .upload-zone .upload-hint {
            color: var(--text-muted);
            font-size: 0.85rem;
        }
        
        .upload-zone .filename {
            color: var(--accent-3);
            font-weight: 500;
            margin-top: 0.5rem;
        }
        
        /* Video preview */
        .video-preview {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            margin-top: 1rem;
        }
        
        .video-preview video {
            width: 100%;
            border-radius: 12px;
            border: 1px solid var(--border);
        }
        
        /* Buttons */
        .btn {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.9rem 2rem;
            border-radius: 12px;
            font-family: inherit;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: var(--transition);
            border: none;
            text-decoration: none;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, var(--accent-1), #7c6cf0);
            color: white;
            box-shadow: 0 4px 15px rgba(108, 92, 231, 0.3);
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(108, 92, 231, 0.4);
        }
        
        .btn-primary:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        
        .btn-secondary {
            background: var(--bg-card);
            color: var(--text-primary);
            border: 1px solid var(--border);
        }
        
        .btn-secondary:hover {
            border-color: var(--accent-2);
        }
        
        .actions {
            display: flex;
            gap: 1rem;
            margin-top: 1.5rem;
            justify-content: center;
        }
        
        /* Progress */
        .progress-container {
            display: none;
            margin-top: 1.5rem;
        }
        
        .progress-container.active { display: block; }
        
        .progress-bar {
            height: 6px;
            background: var(--bg-card);
            border-radius: 3px;
            overflow: hidden;
            margin-bottom: 1rem;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--accent-1), var(--accent-3));
            border-radius: 3px;
            transition: width 0.5s ease;
            width: 0%;
        }
        
        .progress-text {
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.95rem;
        }
        
        .progress-step {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.4rem 0;
            font-size: 0.9rem;
            color: var(--text-secondary);
        }
        
        .progress-step.active { color: var(--accent-3); font-weight: 500; }
        .progress-step.done { color: var(--added-color); }
        .progress-step.done::before { content: '✓ '; }
        .progress-step.active::before { content: '⟳ '; animation: spin 1s linear infinite; }
        
        @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        
        /* Results */
        .results-section { display: none; }
        .results-section.active { display: block; }
        
        .change-list {
            list-style: none;
            padding: 0;
        }
        
        .change-item {
            padding: 1rem 1.2rem;
            margin-bottom: 0.5rem;
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.03);
            border-left: 4px solid;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: var(--transition);
        }
        
        .change-item:hover {
            background: rgba(255, 255, 255, 0.06);
        }
        
        .change-item.moved { border-color: var(--moved-color); }
        .change-item.added { border-color: var(--added-color); }
        .change-item.removed { border-color: var(--removed-color); }
        .change-item.uncertain { border-color: var(--uncertain-color); }
        .change-item.geometric_change { border-color: var(--geometric-color); }
        
        .change-label { font-weight: 500; }
        .change-type {
            font-size: 0.8rem;
            padding: 0.25rem 0.7rem;
            border-radius: 20px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .change-type.moved { background: rgba(255, 107, 53, 0.15); color: var(--moved-color); }
        .change-type.added { background: rgba(46, 204, 113, 0.15); color: var(--added-color); }
        .change-type.removed { background: rgba(231, 76, 60, 0.15); color: var(--removed-color); }
        .change-type.uncertain { background: rgba(149, 165, 166, 0.15); color: var(--uncertain-color); }
        .change-type.geometric_change { background: rgba(52, 152, 219, 0.15); color: var(--geometric-color); }
        
        .confidence-badge {
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-left: 0.5rem;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }
        
        .stat-card {
            text-align: center;
            padding: 1.2rem;
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.03);
        }
        
        .stat-value {
            font-size: 2rem;
            font-weight: 700;
        }
        
        .stat-label {
            font-size: 0.8rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-top: 0.3rem;
        }
        
        /* 3D Viewer */
        .viewer-frame {
            width: 100%;
            height: 500px;
            border: 1px solid var(--border);
            border-radius: var(--radius);
            background: var(--bg-primary);
        }
        
        /* Legend */
        .legend {
            display: flex;
            gap: 1.5rem;
            flex-wrap: wrap;
            margin-top: 1rem;
            padding: 1rem;
            background: rgba(0,0,0,0.2);
            border-radius: 12px;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 0.4rem;
            font-size: 0.85rem;
        }
        
        .legend-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }
        
        /* Empty state */
        .empty-state {
            text-align: center;
            padding: 3rem;
            color: var(--text-muted);
        }
        
        .empty-state .icon { font-size: 3rem; margin-bottom: 1rem; }
        
        /* Responsive */
        @media (max-width: 768px) {
            .upload-grid { grid-template-columns: 1fr; }
            .video-preview { grid-template-columns: 1fr; }
            .logo { font-size: 2rem; }
            .container { padding: 1rem; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo">RoomShift AI</div>
            <p class="subtitle">
                Detect spatial changes between two recordings of the same indoor environment
                using 3D reconstruction and computer vision.
            </p>
        </header>

        <!-- Upload Section -->
        <div class="card" id="uploadCard">
            <h2><span class="icon">📹</span> Upload Recordings</h2>
            <div class="upload-grid">
                <div class="upload-zone" id="beforeZone">
                    <input type="file" id="beforeFile" accept="video/*">
                    <div class="upload-icon">🎬</div>
                    <div class="upload-label">Before Recording</div>
                    <div class="upload-hint">Drag & drop or click to select</div>
                    <div class="filename" id="beforeName"></div>
                </div>
                <div class="upload-zone" id="afterZone">
                    <input type="file" id="afterFile" accept="video/*">
                    <div class="upload-icon">🎬</div>
                    <div class="upload-label">After Recording</div>
                    <div class="upload-hint">Drag & drop or click to select</div>
                    <div class="filename" id="afterName"></div>
                </div>
            </div>
            
            <div class="video-preview" id="videoPreview" style="display:none;">
                <video id="beforeVideo" controls muted></video>
                <video id="afterVideo" controls muted></video>
            </div>

            <div class="actions">
                <button class="btn btn-primary" id="processBtn" disabled onclick="startProcessing()">
                    🔍 Analyze Changes
                </button>
            </div>

            <div class="progress-container" id="progressContainer">
                <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
                <div id="progressSteps">
                    <div class="progress-step" data-step="1">Extracting frames</div>
                    <div class="progress-step" data-step="2">3D Reconstruction</div>
                    <div class="progress-step" data-step="3">Object Detection</div>
                    <div class="progress-step" data-step="4">Scene Alignment</div>
                    <div class="progress-step" data-step="5">Change Detection</div>
                    <div class="progress-step" data-step="6">Generating report</div>
                </div>
            </div>
        </div>

        <!-- Results Section -->
        <div class="results-section" id="resultsSection">
            <div class="card">
                <h2><span class="icon">📊</span> Detection Results</h2>
                <div class="stats-grid" id="statsGrid"></div>
                <ul class="change-list" id="changeList"></ul>
            </div>

            <div class="card">
                <h2><span class="icon">🌐</span> 3D Scene Visualization</h2>
                <iframe class="viewer-frame" id="viewer3d" src="" frameborder="0"></iframe>
                <div class="legend">
                    <div class="legend-item"><div class="legend-dot" style="background:var(--moved-color)"></div> Moved</div>
                    <div class="legend-item"><div class="legend-dot" style="background:var(--added-color)"></div> Added</div>
                    <div class="legend-item"><div class="legend-dot" style="background:var(--removed-color)"></div> Removed</div>
                    <div class="legend-item"><div class="legend-dot" style="background:var(--geometric-color)"></div> Geometric</div>
                    <div class="legend-item"><div class="legend-dot" style="background:var(--uncertain-color)"></div> Uncertain</div>
                </div>
            </div>

            <div class="card">
                <h2><span class="icon">📋</span> Download Report</h2>
                <div class="actions" style="justify-content:flex-start;">
                    <a class="btn btn-secondary" id="downloadJson" href="#" download>📄 JSON Report</a>
                    <a class="btn btn-secondary" id="downloadTxt" href="#" download>📝 Text Summary</a>
                </div>
            </div>
        </div>
    </div>

    <script>
        const beforeInput = document.getElementById('beforeFile');
        const afterInput = document.getElementById('afterFile');
        const processBtn = document.getElementById('processBtn');

        function setupUploadZone(input, zone, nameEl, videoEl) {
            input.addEventListener('change', () => {
                if (input.files.length > 0) {
                    zone.classList.add('has-file');
                    nameEl.textContent = input.files[0].name;
                    const url = URL.createObjectURL(input.files[0]);
                    videoEl.src = url;
                    document.getElementById('videoPreview').style.display = 'grid';
                    checkReady();
                }
            });
            
            zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('dragover'); });
            zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
            zone.addEventListener('drop', (e) => {
                e.preventDefault();
                zone.classList.remove('dragover');
                if (e.dataTransfer.files.length > 0) {
                    input.files = e.dataTransfer.files;
                    input.dispatchEvent(new Event('change'));
                }
            });
        }

        setupUploadZone(beforeInput, document.getElementById('beforeZone'),
                       document.getElementById('beforeName'), document.getElementById('beforeVideo'));
        setupUploadZone(afterInput, document.getElementById('afterZone'),
                       document.getElementById('afterName'), document.getElementById('afterVideo'));

        function checkReady() {
            processBtn.disabled = !(beforeInput.files.length > 0 && afterInput.files.length > 0);
        }

        async function startProcessing() {
            const formData = new FormData();
            formData.append('before', beforeInput.files[0]);
            formData.append('after', afterInput.files[0]);

            processBtn.disabled = true;
            const progress = document.getElementById('progressContainer');
            progress.classList.add('active');

            try {
                const res = await fetch('/api/process', { method: 'POST', body: formData });
                const data = await res.json();
                
                if (data.job_id) {
                    pollStatus(data.job_id);
                } else {
                    alert('Error: ' + (data.error || 'Unknown error'));
                    processBtn.disabled = false;
                }
            } catch (e) {
                alert('Upload failed: ' + e.message);
                processBtn.disabled = false;
            }
        }

        function pollStatus(jobId) {
            const interval = setInterval(async () => {
                try {
                    const res = await fetch('/api/status/' + jobId);
                    const data = await res.json();
                    updateProgress(data.step, data.total_steps);
                    
                    if (data.status === 'done') {
                        clearInterval(interval);
                        showResults(data.result);
                    } else if (data.status === 'error') {
                        clearInterval(interval);
                        alert('Processing error: ' + data.error);
                        processBtn.disabled = false;
                    }
                } catch (e) {
                    clearInterval(interval);
                }
            }, 2000);
        }

        function updateProgress(step, total) {
            const pct = (step / total * 100);
            document.getElementById('progressFill').style.width = pct + '%';
            document.querySelectorAll('.progress-step').forEach(el => {
                const s = parseInt(el.dataset.step);
                el.classList.remove('active', 'done');
                if (s < step) el.classList.add('done');
                else if (s === step) el.classList.add('active');
            });
        }

        function showResults(result) {
            document.getElementById('progressContainer').classList.remove('active');
            const section = document.getElementById('resultsSection');
            section.classList.add('active');

            // Stats
            const stats = result.changes || [];
            const counts = {};
            stats.forEach(c => { counts[c.change_type] = (counts[c.change_type] || 0) + 1; });
            
            const colors = {moved:'#FF6B35',added:'#2ECC71',removed:'#E74C3C',uncertain:'#95A5A6',geometric_change:'#3498DB'};
            let statsHtml = '';
            for (const [type, count] of Object.entries(counts)) {
                statsHtml += `<div class="stat-card">
                    <div class="stat-value" style="color:${colors[type]||'#fff'}">${count}</div>
                    <div class="stat-label">${type.replace('_',' ')}</div>
                </div>`;
            }
            statsHtml += `<div class="stat-card">
                <div class="stat-value">${stats.length}</div>
                <div class="stat-label">Total Changes</div>
            </div>`;
            if (result.alignment_confidence != null) {
                statsHtml += `<div class="stat-card">
                    <div class="stat-value">${(result.alignment_confidence*100).toFixed(0)}%</div>
                    <div class="stat-label">Alignment</div>
                </div>`;
            }
            document.getElementById('statsGrid').innerHTML = statsHtml;

            // Change list
            let listHtml = '';
            stats.forEach(c => {
                listHtml += `<li class="change-item ${c.change_type}">
                    <span>
                        <span class="change-label">${c.object_label || 'Region'}</span>
                        <span class="confidence-badge">${(c.confidence*100).toFixed(0)}%</span>
                    </span>
                    <span class="change-type ${c.change_type}">${c.change_type.replace('_',' ')}</span>
                </li>`;
            });
            document.getElementById('changeList').innerHTML = listHtml || '<li class="empty-state"><div class="icon">🔍</div><p>No changes detected</p></li>';

            // 3D viewer
            if (result.viz_path) {
                document.getElementById('viewer3d').src = '/outputs/' + result.viz_path;
            }

            // Downloads
            if (result.json_path) {
                document.getElementById('downloadJson').href = '/outputs/' + result.json_path;
            }
            if (result.txt_path) {
                document.getElementById('downloadTxt').href = '/outputs/' + result.txt_path;
            }

            section.scrollIntoView({ behavior: 'smooth' });
        }
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/process', methods=['POST'])
def process_videos():
    """Upload and start processing two videos."""
    if 'before' not in request.files or 'after' not in request.files:
        return jsonify({'error': 'Both before and after videos are required'}), 400
    
    before_file = request.files['before']
    after_file = request.files['after']
    
    if not before_file.filename or not after_file.filename:
        return jsonify({'error': 'Invalid file names'}), 400
    
    # Save uploads
    import uuid
    job_id = uuid.uuid4().hex[:8]
    job_dir = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
    os.makedirs(job_dir, exist_ok=True)
    
    before_path = os.path.join(job_dir, 'before.mp4')
    after_path = os.path.join(job_dir, 'after.mp4')
    before_file.save(before_path)
    after_file.save(after_path)
    
    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
    os.makedirs(output_dir, exist_ok=True)
    
    _jobs[job_id] = {
        'status': 'processing',
        'step': 1,
        'total_steps': 7,
        'before_path': before_path,
        'after_path': after_path,
        'output_dir': output_dir,
    }
    
    # Run pipeline in background thread
    import threading
    thread = threading.Thread(target=_run_job, args=(job_id,))
    thread.daemon = True
    thread.start()
    
    return jsonify({'job_id': job_id})


@app.route('/api/status/<job_id>')
def job_status(job_id):
    """Check processing status."""
    if job_id not in _jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = _jobs[job_id]
    return jsonify(job)


@app.route('/outputs/<path:filename>')
def serve_output(filename):
    """Serve output files."""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


def _run_job(job_id: str):
    """Run the pipeline in a background thread."""
    job = _jobs[job_id]
    
    try:
        from roomshift.pipeline import run_pipeline
        from roomshift.config import get_default_config
        
        config = get_default_config()
        
        report = run_pipeline(
            before_video=job['before_path'],
            after_video=job['after_path'],
            output_dir=job['output_dir'],
            config=config,
        )
        
        # Prepare result
        rel_dir = os.path.basename(job['output_dir'])
        job['status'] = 'done'
        job['step'] = 7
        job['result'] = {
            'changes': [c.to_dict() for c in report.changes],
            'alignment_confidence': report.alignment.confidence if report.alignment else None,
            'processing_time': report.processing_time_seconds,
            'viz_path': f"{rel_dir}/scene_3d.html",
            'json_path': f"{rel_dir}/change_report.json",
            'txt_path': f"{rel_dir}/change_summary.txt",
        }
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        job['status'] = 'error'
        job['error'] = str(e)


if __name__ == '__main__':
    # Add src to path
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    print("\n🚀 RoomShift AI — Starting web interface")
    print("   Open http://localhost:5000 in your browser\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
