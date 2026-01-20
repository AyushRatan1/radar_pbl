import { useState, useRef, useEffect } from 'react';

export default function ResultsCard({ results, onReset, onDownload }) {
    const [sliderPosition, setSliderPosition] = useState(50);
    const [activeTab, setActiveTab] = useState('comparison');
    const containerRef = useRef(null);
    const isDragging = useRef(false);

    const handleMouseDown = () => {
        isDragging.current = true;
    };

    const handleMouseUp = () => {
        isDragging.current = false;
    };

    const handleMouseMove = (e) => {
        if (!isDragging.current || !containerRef.current) return;
        const rect = containerRef.current.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const percent = Math.min(100, Math.max(0, (x / rect.width) * 100));
        setSliderPosition(percent);
    };

    useEffect(() => {
        document.addEventListener('mouseup', handleMouseUp);
        document.addEventListener('mousemove', handleMouseMove);
        return () => {
            document.removeEventListener('mouseup', handleMouseUp);
            document.removeEventListener('mousemove', handleMouseMove);
        };
    }, []);

    const scaleFactor = results.enhanced.width / results.original.width;

    return (
        <div className="card">
            <div className="card-header">
                <h3 className="card-title">Enhancement Results</h3>
                <div style={{ display: 'flex', gap: '8px' }}>
                    <button className="btn btn-icon" onClick={onDownload} title="Download">⬇️</button>
                    <button className="btn btn-icon" onClick={onReset} title="New Upload">🔄</button>
                </div>
            </div>

            {/* Tab Navigation */}
            <div className="results-tabs">
                <button
                    className={`tab-btn ${activeTab === 'comparison' ? 'active' : ''}`}
                    onClick={() => setActiveTab('comparison')}
                >
                    📊 Comparison
                </button>
                <button
                    className={`tab-btn ${activeTab === 'pipeline' ? 'active' : ''}`}
                    onClick={() => setActiveTab('pipeline')}
                >
                    ⚙️ Processing Pipeline
                </button>
                <button
                    className={`tab-btn ${activeTab === 'analysis' ? 'active' : ''}`}
                    onClick={() => setActiveTab('analysis')}
                >
                    📈 Analysis
                </button>
            </div>

            <div className="card-body">
                {/* Comparison Tab */}
                {activeTab === 'comparison' && (
                    <>
                        <div
                            className="comparison-wrapper"
                            ref={containerRef}
                            onMouseDown={handleMouseDown}
                        >
                            <div
                                className="comparison-image original"
                                style={{
                                    backgroundImage: `url(data:image/png;base64,${results.original.image})`,
                                    clipPath: `inset(0 ${100 - sliderPosition}% 0 0)`
                                }}
                            >
                                <span className="image-label left">Bicubic</span>
                            </div>
                            <div
                                className="comparison-image enhanced"
                                style={{
                                    backgroundImage: `url(data:image/png;base64,${results.enhanced.image})`,
                                    clipPath: `inset(0 0 0 ${sliderPosition}%)`
                                }}
                            >
                                <span className="image-label right">AI Enhanced</span>
                            </div>
                            <div
                                className="comparison-slider"
                                style={{ left: `${sliderPosition}%` }}
                            />
                        </div>

                        <div className="results-stats">
                            <div className="result-stat">
                                <div className="result-stat-label">Original</div>
                                <div className="result-stat-value">
                                    {results.original.width}×{results.original.height}
                                </div>
                            </div>
                            <div className="result-stat">
                                <div className="result-stat-label">Enhanced</div>
                                <div className="result-stat-value">
                                    {results.enhanced.width}×{results.enhanced.height}
                                </div>
                            </div>
                            <div className="result-stat">
                                <div className="result-stat-label">Time</div>
                                <div className="result-stat-value">
                                    {results.processing_time_ms.toFixed(0)}ms
                                </div>
                            </div>
                        </div>
                    </>
                )}

                {/* Pipeline Tab */}
                {activeTab === 'pipeline' && (
                    <div className="pipeline-view">
                        <h4 className="pipeline-title">Step-by-Step Processing Pipeline</h4>

                        <div className="pipeline-steps">
                            <div className="pipeline-step completed">
                                <div className="step-number">1</div>
                                <div className="step-content">
                                    <h5>Image Upload</h5>
                                    <p>SAR image received from user</p>
                                    <div className="step-detail">
                                        Format: PNG/JPEG • Size: {results.original.width}×{results.original.height}
                                    </div>
                                </div>
                                <div className="step-status">✓</div>
                            </div>

                            <div className="pipeline-connector"></div>

                            <div className="pipeline-step completed">
                                <div className="step-number">2</div>
                                <div className="step-content">
                                    <h5>Preprocessing</h5>
                                    <p>Convert to grayscale & normalize</p>
                                    <div className="step-detail">
                                        Grayscale conversion • Pixel values: 0-1 range
                                    </div>
                                </div>
                                <div className="step-status">✓</div>
                            </div>

                            <div className="pipeline-connector"></div>

                            <div className="pipeline-step completed">
                                <div className="step-number">3</div>
                                <div className="step-content">
                                    <h5>Feature Extraction</h5>
                                    <p>Conv layers extract image features</p>
                                    <div className="step-detail">
                                        Layer 1: 64 filters (5×5) → Layer 2: 64 filters (3×3)
                                    </div>
                                </div>
                                <div className="step-status">✓</div>
                            </div>

                            <div className="pipeline-connector"></div>

                            <div className="pipeline-step completed">
                                <div className="step-number">4</div>
                                <div className="step-content">
                                    <h5>Feature Refinement</h5>
                                    <p>Compress and prepare for upscaling</p>
                                    <div className="step-detail">
                                        Layer 3: 32 filters (3×3) → Layer 4: 16 channels
                                    </div>
                                </div>
                                <div className="step-status">✓</div>
                            </div>

                            <div className="pipeline-connector"></div>

                            <div className="pipeline-step completed">
                                <div className="step-number">5</div>
                                <div className="step-content">
                                    <h5>Sub-Pixel Convolution</h5>
                                    <p>PixelShuffle for 4× upscaling</p>
                                    <div className="step-detail">
                                        Rearrange: H×W×16 → 4H×4W×1 (efficient upscaling)
                                    </div>
                                </div>
                                <div className="step-status">✓</div>
                            </div>

                            <div className="pipeline-connector"></div>

                            <div className="pipeline-step completed">
                                <div className="step-number">6</div>
                                <div className="step-content">
                                    <h5>Output Generation</h5>
                                    <p>Final enhanced image ready</p>
                                    <div className="step-detail">
                                        Output: {results.enhanced.width}×{results.enhanced.height} • Time: {results.processing_time_ms.toFixed(0)}ms
                                    </div>
                                </div>
                                <div className="step-status">✓</div>
                            </div>
                        </div>
                    </div>
                )}

                {/* Analysis Tab */}
                {activeTab === 'analysis' && (
                    <div className="analysis-view">
                        <h4 className="analysis-title">Enhancement Analysis</h4>

                        <div className="analysis-grid">
                            <div className="analysis-card">
                                <div className="analysis-icon">📐</div>
                                <div className="analysis-info">
                                    <h5>Resolution Increase</h5>
                                    <div className="analysis-value">{scaleFactor}× Scale</div>
                                    <p>{results.original.width}×{results.original.height} → {results.enhanced.width}×{results.enhanced.height}</p>
                                </div>
                            </div>

                            <div className="analysis-card">
                                <div className="analysis-icon">🔢</div>
                                <div className="analysis-info">
                                    <h5>Pixel Count</h5>
                                    <div className="analysis-value">{(scaleFactor * scaleFactor)}× More Pixels</div>
                                    <p>{(results.original.width * results.original.height).toLocaleString()} → {(results.enhanced.width * results.enhanced.height).toLocaleString()}</p>
                                </div>
                            </div>

                            <div className="analysis-card">
                                <div className="analysis-icon">⚡</div>
                                <div className="analysis-info">
                                    <h5>Processing Speed</h5>
                                    <div className="analysis-value">{results.processing_time_ms.toFixed(0)}ms</div>
                                    <p>{(1000 / results.processing_time_ms).toFixed(1)} images/second</p>
                                </div>
                            </div>

                            <div className="analysis-card">
                                <div className="analysis-icon">🧠</div>
                                <div className="analysis-info">
                                    <h5>Model Used</h5>
                                    <div className="analysis-value">{results.model || 'ESPCN'}</div>
                                    <p>Efficient Sub-Pixel CNN</p>
                                </div>
                            </div>
                        </div>

                        <div className="analysis-details">
                            <h5>Technical Details</h5>
                            <table className="details-table">
                                <tbody>
                                    <tr>
                                        <td>Input Dimensions</td>
                                        <td>{results.original.width} × {results.original.height} pixels</td>
                                    </tr>
                                    <tr>
                                        <td>Output Dimensions</td>
                                        <td>{results.enhanced.width} × {results.enhanced.height} pixels</td>
                                    </tr>
                                    <tr>
                                        <td>Scale Factor</td>
                                        <td>{scaleFactor}× (horizontal & vertical)</td>
                                    </tr>
                                    <tr>
                                        <td>Processing Time</td>
                                        <td>{results.processing_time_ms.toFixed(2)} milliseconds</td>
                                    </tr>
                                    <tr>
                                        <td>Throughput</td>
                                        <td>{((results.enhanced.width * results.enhanced.height) / results.processing_time_ms * 1000).toFixed(0).toLocaleString()} pixels/sec</td>
                                    </tr>
                                    <tr>
                                        <td>Algorithm</td>
                                        <td>ESPCN (Shi et al., 2016)</td>
                                    </tr>
                                    <tr>
                                        <td>Upscaling Method</td>
                                        <td>Sub-Pixel Convolution (PixelShuffle)</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>

                        <div className="analysis-comparison">
                            <h5>Method Comparison</h5>
                            <div className="comparison-methods">
                                <div className="method-item">
                                    <span className="method-name">Nearest Neighbor</span>
                                    <div className="method-bar">
                                        <div className="method-fill" style={{ width: '30%' }}></div>
                                    </div>
                                    <span className="method-label">Low Quality</span>
                                </div>
                                <div className="method-item">
                                    <span className="method-name">Bilinear</span>
                                    <div className="method-bar">
                                        <div className="method-fill" style={{ width: '50%' }}></div>
                                    </div>
                                    <span className="method-label">Medium</span>
                                </div>
                                <div className="method-item">
                                    <span className="method-name">Bicubic</span>
                                    <div className="method-bar">
                                        <div className="method-fill" style={{ width: '65%' }}></div>
                                    </div>
                                    <span className="method-label">Good</span>
                                </div>
                                <div className="method-item highlight">
                                    <span className="method-name">ESPCN (AI)</span>
                                    <div className="method-bar">
                                        <div className="method-fill ai" style={{ width: '90%' }}></div>
                                    </div>
                                    <span className="method-label">Excellent</span>
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                <div className="actions-bar">
                    <button className="btn btn-secondary" onClick={onReset}>Upload New</button>
                    <button className="btn btn-primary" onClick={onDownload}>⬇️ Download Enhanced</button>
                </div>
            </div>
        </div>
    );
}
