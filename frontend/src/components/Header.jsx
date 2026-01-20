export default function Header({ currentPage }) {
    const titles = {
        dashboard: { title: 'SAR Image Enhancement', subtitle: 'Super-resolution for automotive radar imagery' },
        upload: { title: 'Upload Image', subtitle: 'Upload a SAR image for enhancement' },
        history: { title: 'Processing History', subtitle: 'View your previously enhanced images' },
        settings: { title: 'Settings', subtitle: 'Configure model and application settings' },
    };

    const { title, subtitle } = titles[currentPage] || titles.dashboard;

    return (
        <header className="header">
            <div className="header-left">
                <h1>{title}</h1>
                <p>{subtitle}</p>
            </div>
            <div className="header-right">
                <a
                    href="http://localhost:8000/docs"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="btn btn-secondary"
                >
                    API Documentation
                </a>
            </div>
        </header>
    );
}
