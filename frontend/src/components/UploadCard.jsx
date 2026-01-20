import { useCallback, useState, useEffect } from 'react';

export default function UploadCard({
    selectedFile,
    preview,
    isProcessing,
    error,
    onFileSelect,
    onEnhance,
    onReset
}) {
    const [processingStep, setProcessingStep] = useState(0);

    useEffect(() => {
        if (isProcessing) {
            setProcessingStep(1);
            const timers = [
                setTimeout(() => setProcessingStep(2), 300),
                setTimeout(() => setProcessingStep(3), 800),
                setTimeout(() => setProcessingStep(4), 1500),
                setTimeout(() => setProcessingStep(5), 2200),
            ];
            return () => timers.forEach(t => clearTimeout(t));
        } else {
            setProcessingStep(0);
        }
    }, [isProcessing]);

    const handleDrop = useCallback((e) => {
        e.preventDefault();
        e.currentTarget.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            onFileSelect(file);
        }
    }, [onFileSelect]);

    const handleDragOver = (e) => {
        e.preventDefault();
        e.currentTarget.classList.add('drag-over');
    };

    const handleDragLeave = (e) => {
        e.currentTarget.classList.remove('drag-over');
    };

    const handleFileInput = (e) => {
        const file = e.target.files[0];
        if (file) onFileSelect(file);
    };

    const formatFileSize = (bytes) => {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    };

    const processingSteps = [
        { label: 'Uploading image...', icon: '📤' },
        { label: 'Preprocessing (Grayscale)', icon: '🔲' },
        { label: 'Feature Extraction (Conv Layers)', icon: '🧠' },
        { label: 'Super-Resolution (PixelShuffle)', icon: '✨' },
        { label: 'Generating output...', icon: '🖼️' },
    ];

    if (isProcessing) {
        return (
            <div className="card">
                <div className="card-header">
                    <h3 className="card-title">Processing Image</h3>
                </div>
                <div className="card-body">
                    <div className="processing-pipeline">
                        <div className="processing-header">
                            <div className="spinner"></div>
                            <h4>Enhancing with ESPCN Model</h4>
                        </div>

                        <div className="processing-steps-list">
                            {processingSteps.map((step, idx) => (
                                <div
                                    key={idx}
                                    className={`processing-step-item ${idx + 1 < processingStep ? 'completed' :
                                            idx + 1 === processingStep ? 'active' : ''
                                        }`}
                                >
                                    <div className="step-indicator">
                                        {idx + 1 < processingStep ? '✓' : step.icon}
                                    </div>
                                    <span className="step-label">{step.label}</span>
                                    {idx + 1 === processingStep && (
                                        <div className="step-loading">
                                            <div className="loading-dots">
                                                <span></span><span></span><span></span>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>

                        <div className="processing-info">
                            <p>Running on: <strong>CPU</strong></p>
                            <p>Model: <strong>ESPCN (4× upscale)</strong></p>
                        </div>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="card">
            <div className="card-header">
                <h3 className="card-title">Upload SAR Image</h3>
                {selectedFile && (
                    <button className="btn btn-icon" onClick={onReset} title="Clear">✕</button>
                )}
            </div>
            <div className="card-body">
                {error && (
                    <div className="error-banner">⚠️ {error}</div>
                )}

                {!selectedFile ? (
                    <div
                        className="upload-zone"
                        onDrop={handleDrop}
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        onClick={() => document.getElementById('file-input').click()}
                    >
                        <input
                            type="file"
                            id="file-input"
                            accept="image/*"
                            onChange={handleFileInput}
                            style={{ display: 'none' }}
                        />
                        <div className="upload-content">
                            <div className="upload-icon">📤</div>
                            <h3 className="upload-title">Drop your image here</h3>
                            <p className="upload-subtitle">or click to browse files</p>
                            <div className="upload-formats">
                                <span className="format-tag">PNG</span>
                                <span className="format-tag">JPEG</span>
                                <span className="format-tag">TIFF</span>
                            </div>
                        </div>
                    </div>
                ) : (
                    <div className="upload-zone has-file">
                        <div className="file-preview">
                            <img src={preview} alt="Preview" />
                            <div className="file-info">
                                <h4>{selectedFile.name}</h4>
                                <p>{formatFileSize(selectedFile.size)}</p>
                            </div>
                        </div>
                    </div>
                )}

                {selectedFile && (
                    <div className="actions-bar">
                        <button className="btn btn-secondary" onClick={onReset}>Cancel</button>
                        <button className="btn btn-primary" onClick={onEnhance}>🚀 Enhance Image</button>
                    </div>
                )}
            </div>
        </div>
    );
}
