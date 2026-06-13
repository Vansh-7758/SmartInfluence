let currentInfluencers = { main: [], hidden: [] };
let currentBrand = "Brand";
let roster = JSON.parse(localStorage.getItem('smartinfluence_roster')) || [];

function saveRoster() {
    localStorage.setItem('smartinfluence_roster', JSON.stringify(roster));
    updateRosterBadge();
    if (document.getElementById('roster-tab').classList.contains('active')) {
        renderRoster();
    }
}

function updateRosterBadge() {
    const badge = document.getElementById('roster-count');
    if (badge) {
        badge.textContent = roster.length;
        badge.style.display = roster.length > 0 ? 'inline-block' : 'none';
    }
}

function toggleRoster(inf) {
    const index = roster.findIndex(r => r.ACCOUNTNAME === inf.ACCOUNTNAME);
    if (index >= 0) {
        roster.splice(index, 1);
    } else {
        roster.push(inf);
    }
    saveRoster();
}

function isSaved(accountName) {
    return roster.some(r => r.ACCOUNTNAME === accountName);
}

// Helpers for Premium Visual Accents
function getInitials(name) {
    if (!name) return 'CR';
    return name.split(' ').map(w => w[0]).join('').substring(0, 2).toUpperCase();
}

function getNicheClass(niche) {
    if (!niche) return 'grad-default';
    const n = niche.toLowerCase();
    if (n.includes('fit') || n.includes('gym') || n.includes('sport') || n.includes('health')) return 'grad-fitness';
    if (n.includes('fash') || n.includes('style') || n.includes('dress') || n.includes('apparel') || n.includes('wear')) return 'grad-fashion';
    if (n.includes('beaut') || n.includes('makeup') || n.includes('cosm') || n.includes('skin')) return 'grad-beauty';
    if (n.includes('travel') || n.includes('wander') || n.includes('explore') || n.includes('hotel')) return 'grad-travel';
    if (n.includes('food') || n.includes('eat') || n.includes('cook') || n.includes('chef') || n.includes('recip')) return 'grad-food';
    if (n.includes('tech') || n.includes('code') || n.includes('gadg') || n.includes('dev') || n.includes('soft')) return 'grad-tech';
    if (n.includes('life') || n.includes('vlog') || n.includes('personal') || n.includes('blog')) return 'grad-lifestyle';
    return 'grad-default';
}

// Render skeleton shimmer placeholder screens during loading
function renderSkeletons(container, count = 4) {
    container.innerHTML = '';
    for (let i = 0; i < count; i++) {
        const skeleton = document.createElement('div');
        skeleton.className = 'influencer-card skeleton-card card-animate';
        skeleton.style.animationDelay = `${i * 0.08}s`;
        
        skeleton.innerHTML = `
            <div class="card-avatar-wrapper">
                <div class="sk-avatar shimmer-anim"></div>
                <div style="flex:1;">
                    <div class="sk-line name shimmer-anim"></div>
                    <div class="sk-line handle shimmer-anim"></div>
                </div>
            </div>
            <div class="sk-line tag shimmer-anim"></div>
            <div class="sk-metrics">
                <div class="sk-metric shimmer-anim"></div>
                <div class="sk-metric shimmer-anim"></div>
            </div>
            <div class="sk-line button shimmer-anim"></div>
        `;
        container.appendChild(skeleton);
    }
}

// Initialize on DOM Load
document.addEventListener('DOMContentLoaded', () => {
    updateRosterBadge();
});

// Tab Logic
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
        const targetBtn = e.currentTarget;
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        
        targetBtn.classList.add('active');
        document.getElementById(targetBtn.dataset.tab).classList.add('active');
        
        if (targetBtn.dataset.tab === 'roster-tab') {
            renderRoster();
        }
    });
});

// Form Submission
document.getElementById('predict-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const btn = document.getElementById('submit-btn');
    const resultsSection = document.getElementById('results-section');
    
    btn.classList.add('loading');
    btn.disabled = true;
    
    currentBrand = document.getElementById('brand_name').value;

    const payload = {
        brand_name: currentBrand,
        niches: document.getElementById('niches').value,
        brand_desc: document.getElementById('brand_desc').value,
        top_n: parseInt(document.getElementById('top_n').value)
    };

    // Instantly trigger shimmer skeletons to indicate predictive model pipeline running
    resultsSection.classList.remove('hidden');
    renderSkeletons(document.getElementById('main-results'), 4);
    renderSkeletons(document.getElementById('hidden-results'), 2);
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) throw new Error('API Error');
        const data = await response.json();
        
        currentInfluencers = data;
        
        // Wait briefly for a natural UI transition from skeletons to content
        setTimeout(() => {
            renderGrid(document.getElementById('main-results'), data.main);
            renderGrid(document.getElementById('hidden-results'), data.hidden);
        }, 600);

    } catch (error) {
        alert('An error occurred while fetching predictions.');
        console.error(error);
        resultsSection.classList.add('hidden');
    } finally {
        btn.classList.remove('loading');
        btn.disabled = false;
    }
});

function renderGrid(container, influencers) {
    container.innerHTML = '';
    
    if (!influencers || influencers.length === 0) {
        container.innerHTML = `<div class="empty-state">No matching influencers found in this category.</div>`;
        return;
    }

    influencers.forEach((inf, index) => {
        const engRate = parseFloat(inf.INSTAGRAM_ENGAGEMENT_RATE || 0).toFixed(2);
        const followers = parseInt(inf.INSTAGRAM_FOLLOWER_COUNT || 0).toLocaleString();
        const conf = parseFloat(inf.confidence || 0).toFixed(1);
        
        const initials = getInitials(inf.NAME || inf.ACCOUNTNAME);
        const gradClass = getNicheClass(inf.niche || '');
        
        // Format individual tag pills
        let nicheBadges = '';
        if (inf.niche) {
            const list = inf.niche.split(',').map(n => n.trim()).filter(Boolean).slice(0, 3);
            nicheBadges = list.map(n => `<span class="niche-tag">${n}</span>`).join('');
        } else {
            nicheBadges = `<span class="niche-tag">Creator</span>`;
        }

        const card = document.createElement('div');
        card.className = 'influencer-card';
        card.style.animationDelay = `${index * 0.05}s`;
        
        const btnText = isSaved(inf.ACCOUNTNAME) ? '✓ Shortlisted' : '+ Shortlist Creator';
        const btnClass = isSaved(inf.ACCOUNTNAME) ? 'btn-action btn-add-roster saved' : 'btn-action btn-add-roster';
        
        card.innerHTML = `
            <div class="card-avatar-wrapper">
                <div class="card-avatar ${gradClass}">${initials}</div>
                <div class="card-meta-info">
                    <div class="card-name-wrap">
                        <div class="card-name" title="${inf.NAME || inf.ACCOUNTNAME}">${inf.NAME || inf.ACCOUNTNAME}</div>
                        <svg width="14" height="14" fill="currentColor" viewBox="0 0 24 24" class="card-verified-icon" title="Verified Creator">
                            <path d="M12 2C6.5 2 2 6.5 2 12s4.5 10 10 10 10-4.5 10-10S17.5 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
                        </svg>
                    </div>
                    <div class="card-username">@${inf.ACCOUNTNAME}</div>
                </div>
            </div>
            
            <div class="niche-tags">
                ${nicheBadges}
            </div>

            <div class="metric-group">
                <div class="metric">
                    <div class="metric-icon">
                        <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24">
                            <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
                            <circle cx="9" cy="7" r="4"></circle>
                            <path d="M23 21v-2a4 4 0 0 0-3-3.87"></path>
                            <path d="M16 3.13a4 4 0 0 1 0 7.75"></path>
                        </svg>
                    </div>
                    <div class="metric-details">
                        <div class="metric-label">Followers</div>
                        <div class="metric-value">${followers}</div>
                    </div>
                </div>
                <div class="metric">
                    <div class="metric-icon">
                        <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24">
                            <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"></path>
                        </svg>
                    </div>
                    <div class="metric-details">
                        <div class="metric-label">Engagement</div>
                        <div class="metric-value">${engRate}%</div>
                    </div>
                </div>
            </div>
            
            <div class="card-footer">
                <div style="flex:1; margin-bottom: 0.85rem;">
                    <div class="confidence-header">
                        <span>Predictive Fit Score</span>
                        <span>${conf}%</span>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: 0%"></div>
                    </div>
                </div>
                <button class="${btnClass}">${btnText}</button>
            </div>
        `;
        
        const addBtn = card.querySelector('.btn-add-roster');
        addBtn.addEventListener('click', (ev) => {
            if (isSaved(inf.ACCOUNTNAME)) return;
            toggleRoster(inf);
            ev.currentTarget.textContent = '✓ Shortlisted';
            ev.currentTarget.className = 'btn-action btn-add-roster saved';
        });
        
        container.appendChild(card);
        
        requestAnimationFrame(() => {
            card.classList.add('card-animate');
            setTimeout(() => {
                const bar = card.querySelector('.confidence-fill');
                if (bar) bar.style.width = `${conf}%`;
            }, index * 40 + 200);
        });
    });
}

function renderRoster() {
    const container = document.getElementById('roster-results');
    if (!container) return;
    container.innerHTML = '';
    
    let totalReach = 0;
    let totalEng = 0;
    
    roster.forEach(inf => {
        totalReach += parseInt(inf.INSTAGRAM_FOLLOWER_COUNT || 0);
        totalEng += parseFloat(inf.INSTAGRAM_ENGAGEMENT_RATE || 0);
    });
    
    const avgEng = roster.length > 0 ? (totalEng / roster.length).toFixed(2) : '0.00';
    
    document.getElementById('roster-total-count').textContent = roster.length;
    document.getElementById('roster-total-reach').textContent = totalReach.toLocaleString();
    document.getElementById('roster-avg-eng').textContent = avgEng + '%';
    
    if (roster.length === 0) {
        container.innerHTML = `<div class="empty-state">Your roster is empty. Go to the Discovery Tool to add influencers!</div>`;
        return;
    }
    
    roster.forEach((inf, index) => {
        const engRate = parseFloat(inf.INSTAGRAM_ENGAGEMENT_RATE || 0).toFixed(2);
        const followers = parseInt(inf.INSTAGRAM_FOLLOWER_COUNT || 0).toLocaleString();
        
        const initials = getInitials(inf.NAME || inf.ACCOUNTNAME);
        const gradClass = getNicheClass(inf.niche || '');
        
        let nicheBadges = '';
        if (inf.niche) {
            const list = inf.niche.split(',').map(n => n.trim()).filter(Boolean).slice(0, 3);
            nicheBadges = list.map(n => `<span class="niche-tag">${n}</span>`).join('');
        } else {
            nicheBadges = `<span class="niche-tag">Creator</span>`;
        }

        const card = document.createElement('div');
        card.className = 'influencer-card card-animate';
        card.style.animationDelay = `${index * 0.05}s`;
        
        card.innerHTML = `
            <div class="card-avatar-wrapper">
                <div class="card-avatar ${gradClass}">${initials}</div>
                <div class="card-meta-info">
                    <div class="card-name-wrap">
                        <div class="card-name" title="${inf.NAME || inf.ACCOUNTNAME}">${inf.NAME || inf.ACCOUNTNAME}</div>
                        <svg width="14" height="14" fill="currentColor" viewBox="0 0 24 24" class="card-verified-icon" title="Verified Creator">
                            <path d="M12 2C6.5 2 2 6.5 2 12s4.5 10 10 10 10-4.5 10-10S17.5 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
                        </svg>
                    </div>
                    <div class="card-username">@${inf.ACCOUNTNAME}</div>
                </div>
            </div>
            
            <div class="niche-tags">
                ${nicheBadges}
            </div>

            <div class="metric-group">
                <div class="metric">
                    <div class="metric-icon">
                        <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24">
                            <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
                            <circle cx="9" cy="7" r="4"></circle>
                            <path d="M23 21v-2a4 4 0 0 0-3-3.87"></path>
                            <path d="M16 3.13a4 4 0 0 1 0 7.75"></path>
                        </svg>
                    </div>
                    <div class="metric-details">
                        <div class="metric-label">Followers</div>
                        <div class="metric-value">${followers}</div>
                    </div>
                </div>
                <div class="metric">
                    <div class="metric-icon">
                        <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24">
                            <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"></path>
                        </svg>
                    </div>
                    <div class="metric-details">
                        <div class="metric-label">Engagement</div>
                        <div class="metric-value">${engRate}%</div>
                    </div>
                </div>
            </div>
            
            <button class="btn-action btn-remove-roster">
                <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24">
                    <line x1="18" y1="6" x2="6" y2="18"></line>
                    <line x1="6" y1="6" x2="18" y2="18"></line>
                </svg>
                Remove from Roster
            </button>
        `;
        
        card.querySelector('.btn-remove-roster').addEventListener('click', () => {
            toggleRoster(inf);
        });
        
        container.appendChild(card);
    });
}

const clearBtn = document.getElementById('clear-roster-btn');
if (clearBtn) {
    clearBtn.addEventListener('click', () => {
        if(confirm('Are you sure you want to clear your entire roster?')) {
            roster = [];
            saveRoster();
        }
    });
}

// PDF Export Feature
document.getElementById('export-pdf-btn').addEventListener('click', () => {
    if (!window.jspdf || !window.jspdf.jsPDF) {
        alert("PDF library is still loading. Please try again.");
        return;
    }

    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();
    
    doc.setFont("helvetica", "bold");
    doc.setFontSize(20);
    doc.text(`SmartInfluence Report: ${currentBrand}`, 14, 20);
    
    doc.setFontSize(12);
    doc.setFont("helvetica", "normal");
    doc.text(`Generated on: ${new Date().toLocaleDateString()}`, 14, 28);

    const generateTable = (title, data, startY) => {
        if (!data || data.length === 0) return startY;
        
        doc.setFontSize(14);
        doc.setFont("helvetica", "bold");
        doc.text(title, 14, startY);
        
        const tableData = data.map(inf => [
            inf.NAME || inf.ACCOUNTNAME,
            `@${inf.ACCOUNTNAME}`,
            parseInt(inf.INSTAGRAM_FOLLOWER_COUNT).toLocaleString(),
            `${parseFloat(inf.INSTAGRAM_ENGAGEMENT_RATE).toFixed(2)}%`,
            `${parseFloat(inf.confidence).toFixed(1)}%`
        ]);

        doc.autoTable({
            startY: startY + 5,
            head: [['Name', 'Handle', 'Followers', 'Engagement', 'Confidence']],
            body: tableData,
            theme: 'striped',
            headStyles: { fillColor: [37, 99, 235] }
        });
        
        return doc.lastAutoTable.finalY + 15;
    };

    let y = 40;
    y = generateTable("Top Recommendations (High Performers)", currentInfluencers.main, y);
    generateTable("Alternatives (High Engagement)", currentInfluencers.hidden, y);

    doc.save(`SmartInfluence_Report_${currentBrand}.pdf`);
});
