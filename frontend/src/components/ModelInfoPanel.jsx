export default function ModelInfoPanel() {
    return (
        <div className="card" style={{ marginTop: '16px' }}>
            <div className="card-header">
                <h3 className="card-title">Model Info</h3>
            </div>
            <div className="card-body">
                <div className="model-info">
                    <div className="model-card">
                        <div className="model-card-header">
                            <span className="model-badge">Active</span>
                            <span className="model-name">ESPCN</span>
                        </div>
                        <p className="model-description">
                            Efficient Sub-Pixel Convolutional Neural Network for real-time image super-resolution.
                        </p>
                        <div className="model-specs">
                            <div className="spec-item">
                                <div className="spec-label">Scale</div>
                                <div className="spec-value">4x</div>
                            </div>
                            <div className="spec-item">
                                <div className="spec-label">Type</div>
                                <div className="spec-value">CNN</div>
                            </div>
                            <div className="spec-item">
                                <div className="spec-label">Channels</div>
                                <div className="spec-value">Grayscale</div>
                            </div>
                            <div className="spec-item">
                                <div className="spec-label">Framework</div>
                                <div className="spec-value">PyTorch</div>
                            </div>
                        </div>
                    </div>

                    <div className="architecture-flow">
                        <div className="arch-node input">Input</div>
                        <div className="arch-arrow">→</div>
                        <div className="arch-node conv">Conv64</div>
                        <div className="arch-arrow">→</div>
                        <div className="arch-node conv">Conv32</div>
                        <div className="arch-arrow">→</div>
                        <div className="arch-node shuffle">Shuffle</div>
                        <div className="arch-arrow">→</div>
                        <div className="arch-node output">Output</div>
                    </div>
                </div>
            </div>
        </div>
    );
}
