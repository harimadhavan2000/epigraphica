document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('uploadForm');
    const status = document.getElementById('status');

    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        status.textContent = 'Processing...';

        const formData = new FormData(uploadForm);
        
        try {
            const response = await fetch('/process', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.success) {
                window.location.href = `/static/output/${data.output_id}.html`;
            } else {
                status.textContent = `Error: ${data.error}`;
            }
        } catch (error) {
            status.textContent = `Error: ${error.message}`;
        }
    });
});

// Navigation functions
function initializeNavigation() {
    let currentPage = 0;
    const pages = document.querySelectorAll('.page');
    
    function navigateTo(pageNum) {
        pages[currentPage].style.display = 'none';
        currentPage = pageNum;
        pages[currentPage].style.display = 'flex';
    }
    
    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft' && currentPage > 0) {
            navigateTo(currentPage - 1);
        } else if (e.key === 'ArrowRight' && currentPage < pages.length - 1) {
            navigateTo(currentPage + 1);
        }
    });
} 