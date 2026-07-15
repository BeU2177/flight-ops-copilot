document.addEventListener('DOMContentLoaded', () => {
    const queryForm = document.getElementById('query-form');
    const queryInput = document.getElementById('query-input');
    const resultsCard = document.getElementById('results-card');
    const loader = document.getElementById('loader');
    const outputText = document.getElementById('output-text');
    const intentTag = document.getElementById('intent-tag');
    const btnCopy = document.getElementById('btn-copy');
    
    // Agent Badges
    const badges = {
        'DecisionAgent': document.getElementById('badge-decision'),
        'WeatherAgent': document.getElementById('badge-weather'),
        'RAGAgent': document.getElementById('badge-rag'),
        'PerformanceAgent': document.getElementById('badge-perf')
    };

    // Preset buttons query injection
    document.querySelectorAll('.btn-preset').forEach(button => {
        button.addEventListener('click', () => {
            const query = button.getAttribute('data-query');
            queryInput.value = query;
            queryInput.focus();
            // Scroll to form input area
            queryForm.scrollIntoView({ behavior: 'smooth' });
        });
    });

    // Reset all badges to inactive
    function resetBadges() {
        Object.values(badges).forEach(badge => badge.classList.remove('active'));
    }

    // Activate specific agent badge
    function activateBadge(agentName) {
        resetBadges();
        if (badges[agentName]) {
            badges[agentName].classList.add('active');
        } else {
            badges['DecisionAgent'].classList.add('active');
        }
    }

    // Simple parser to format Markdown strings into HTML
    function formatMarkdown(text) {
        if (!text) return "";
        
        let lines = text.split('\n');
        let inList = false;
        let htmlLines = [];

        lines.forEach(line => {
            let processedLine = line;

            // 1. Convert headers (e.g., ### Title)
            if (processedLine.startsWith('###')) {
                if (inList) {
                    htmlLines.push('</ul>');
                    inList = false;
                }
                let headerText = processedLine.replace(/^###\s*/, '');
                htmlLines.push(`<h3>${headerText}</h3>`);
                return;
            }

            // 2. Convert Horizontal Lines (e.g., ---)
            if (processedLine.trim() === '---') {
                if (inList) {
                    htmlLines.push('</ul>');
                    inList = false;
                }
                htmlLines.push('<hr class="header-line">');
                return;
            }

            // 3. Convert List Items (e.g., - Item or * Item)
            let listMatch = processedLine.match(/^[-*]\s+(.*)/);
            if (listMatch) {
                if (!inList) {
                    htmlLines.push('<ul>');
                    inList = true;
                }
                let itemContent = listMatch[1];
                // Bold items
                itemContent = itemContent.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
                // Citation formatting [Source: B738-FCOM-Rev21...]
                itemContent = itemContent.replace(/\[Source:\s*(.*?)\]/g, '<span class="citation-tag">Source: $1</span>');
                htmlLines.push(`<li>${itemContent}</li>`);
                return;
            }

            // If we are in a list but the line is not a list item, close the list
            if (inList && processedLine.trim() !== '') {
                htmlLines.push('</ul>');
                inList = false;
            }

            // 4. Bold formatting in regular lines
            processedLine = processedLine.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            
            // 5. Citation tags in regular lines
            processedLine = processedLine.replace(/\[Source:\s*(.*?)\]/g, '<span class="citation-tag">Source: $1</span>');

            if (processedLine.trim() !== '') {
                htmlLines.push(`<p>${processedLine}</p>`);
            }
        });

        if (inList) {
            htmlLines.push('</ul>');
        }

        return htmlLines.join('\n');
    }

    // Submit Query form
    queryForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = queryInput.value.trim();
        if (!query) return;

        // Show results card and loading state
        resultsCard.classList.remove('hidden');
        loader.classList.remove('hidden');
        outputText.classList.add('hidden');
        outputText.innerHTML = '';
        intentTag.innerText = 'ROUTING QUERY...';
        resetBadges();
        badges['DecisionAgent'].classList.add('active'); // Start with Decision Agent / Orchestrator active

        // Scroll output into view
        resultsCard.scrollIntoView({ behavior: 'smooth' });

        try {
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: query })
            });

            if (!response.ok) {
                throw new Error(`Server returned HTTP error ${response.status}`);
            }

            const data = await response.json();
            
            if (data.status === 'success') {
                // Update intent tag and agent badges
                intentTag.innerText = `INTENT: ${data.intent}`;
                activateBadge(data.agent);
                
                // Format and render output
                outputText.innerHTML = formatMarkdown(data.response);
            } else {
                intentTag.innerText = 'EXECUTION ERROR';
                outputText.innerHTML = `<div class="alert-card danger"><p><strong>Error executing command:</strong> ${data.message || 'Unknown error occurred'}</p></div>`;
            }
        } catch (error) {
            console.error('API Error:', error);
            intentTag.innerText = 'NETWORK CONNECTION FAILURE';
            outputText.innerHTML = `<div class="alert-card danger"><p><strong>Network Error:</strong> Failed to connect to the backend server. Make sure server.py is running on port 8000.</p></div>`;
        } finally {
            loader.classList.add('hidden');
            outputText.classList.remove('hidden');
        }
    });

    // Copy to clipboard
    btnCopy.addEventListener('click', () => {
        const text = outputText.innerText;
        navigator.clipboard.writeText(text).then(() => {
            const originalText = btnCopy.innerText;
            btnCopy.innerText = 'Copied!';
            btnCopy.style.background = 'rgba(0, 230, 118, 0.2)';
            btnCopy.style.borderColor = 'var(--success)';
            setTimeout(() => {
                btnCopy.innerText = originalText;
                btnCopy.style.background = '';
                btnCopy.style.borderColor = '';
            }, 2000);
        }).catch(err => {
            console.error('Failed to copy text: ', err);
        });
    });
});
