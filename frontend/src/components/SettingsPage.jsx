import { useState, useEffect } from 'react';

export default function SettingsPage() {
    const [modelInfo, setModelInfo] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetch('http://localhost:8000/model/info')
            .then(res => res.json())
            .then(data => {
                setModelInfo(data);
                setLoading(false);
            })
            .catch(() => setLoading(false));
    }, []);

    return (
        <div className="settings-page">
            <div className="card">
                <div className="card-header">
                    <h3 className="card-title">Model Configuration</h3>
                </div>
                <div className="card-body">
                    {loading ? (
                        <div className="empty-state">
                            <div className="spinner"></div>
                            <p>Loading model info...</p>
                        </div>
                    ) : modelInfo ? (
                        <div className="settings-grid">
                            <div className="setting-item">
                                <label>Model Name</label>
                                <div className="setting-value">{modelInfo.model_name}</div>
                            </div>
                            <div className="setting-item">
                                <label>Architecture</label>
                                <div className="setting-value">{modelInfo.architecture}</div>
                            </div>
                            <div className="setting-item">
                                <label>Scale Factor</label>
                                <div className="setting-value">{modelInfo.scale_factor}x</div>
                            </div>
                            <div className="setting-item">
                                <label>Input Channels</label>
                                <div className="setting-value">{modelInfo.input_channels} (Grayscale)</div>
                            </div>
                            <div className="setting-item full-width">
                                <label>Description</label>
                                <div className="setting-value">{modelInfo.description}</div>
                            </div>
                            <div className="setting-item full-width">
                                <label>Benefits</label>
                                <ul className="benefits-list">
                                    {modelInfo.benefits?.map((benefit, i) => (
                                        <li key={i}>{benefit}</li>
                                    ))}
                                </ul>
                            </div>
                        </div>
                    ) : (
                        <div className="empty-state">
                            <p>Could not load model info</p>
                        </div>
                    )}
                </div>
            </div>

            <div className="card" style={{ marginTop: '16px' }}>
                <div className="card-header">
                    <h3 className="card-title">Training Status</h3>
                </div>
                <div className="card-body">
                    <div className="training-status">
                        <div className="status-indicator warning">
                            <span className="status-icon">⚠️</span>
                            <div>
                                <h4>Using Random Weights</h4>
                                <p>Model is initialized with random weights. Run the training script to train on SAR data.</p>
                            </div>
                        </div>
                        <div className="training-instructions">
                            <h4>To train the model:</h4>
                            <pre><code>cd backend{'\n'}python train.py</code></pre>
                            <p>This will download SAR datasets and train the ESPCN model.</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
