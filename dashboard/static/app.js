/* ═══════════════════════════════════════════════════════════════════
   AEGIS — Client Application
   ═══════════════════════════════════════════════════════════════════ */

const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

// ═══════════════════════════════════════════════════════════════════
// API CLIENT
// ═══════════════════════════════════════════════════════════════════

async function api(path) {
    const res = await fetch(path);
    if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error || `HTTP ${res.status}`);
    }
    return res.json();
}

async function apiPost(path, body) {
    const res = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    return res.json();
}

// ═══════════════════════════════════════════════════════════════════
// FORMATTERS
// ═══════════════════════════════════════════════════════════════════

function fmtNum(n, prefix = "") {
    if (n == null || n === 0) return `${prefix}0`;
    const a = Math.abs(n);
    if (a >= 1e12) return `${prefix}${(n / 1e12).toFixed(2)}T`;
    if (a >= 1e9) return `${prefix}${(n / 1e9).toFixed(2)}B`;
    if (a >= 1e6) return `${prefix}${(n / 1e6).toFixed(2)}M`;
    if (a >= 1e3) return `${prefix}${(n / 1e3).toFixed(1)}K`;
    return `${prefix}${n.toFixed(2)}`;
}

function fmtPct(n) {
    if (n == null) return "—";
    return `${n >= 0 ? "+" : ""}${n.toFixed(2)}%`;
}

function fmtPrice(n) {
    if (n == null || n === 0) return "—";
    return n.toLocaleString(undefined, {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    });
}

// ═══════════════════════════════════════════════════════════════════
// DOM HELPERS
// ═══════════════════════════════════════════════════════════════════

function loading() {
    return '<div class="loading-wrap"><div class="spinner"></div> Loading…</div>';
}

function emptyState(msg, hint) {
    return `<div class="empty-state"><p>${msg}</p>${
        hint ? `<code>${hint}</code>` : ""
    }</div>`;
}

function metricGrid(items) {
    return `<div class="metric-grid">${items
        .map(
            (i) => `
        <div class="metric-cell">
            <div class="metric-label">${i.label}</div>
            <div class="metric-value">${i.value}</div>
            ${
                i.delta != null
                    ? `<div class="metric-delta ${
                          String(i.delta).startsWith("-")
                              ? "delta-down"
                              : "delta-up"
                      }">${i.delta}</div>`
                    : ""
            }
        </div>`
        )
        .join("")}</div>`;
}

function pageHeader(title, subtitle, badges = []) {
    const badgeHtml = badges
        .map(
            (b) =>
                `<span class="badge badge-${b.type}">${
                    b.dot
                        ? `<span class="status-dot ${
                              b.dotClass || "open"
                          }"></span>`
                        : ""
                }${b.label}</span>`
        )
        .join("");
    return `
        <div class="page-header">
            <div>
                <h1>${title}</h1>
                ${subtitle ? `<p>${subtitle}</p>` : ""}
            </div>
            <div class="header-badges">${badgeHtml}</div>
        </div>`;
}

// ═══════════════════════════════════════════════════════════════════
// CHART CONSTANTS
// ═══════════════════════════════════════════════════════════════════

const C = {
    green: "#10b981",
    red: "#f43f5e",
    indigo: "#6366f1",
    sky: "#38bdf8",
    amber: "#f59e0b",
    bg: "rgba(0,0,0,0)",
    plotBg: "rgba(10,10,15,0.95)",
    grid: "rgba(255,255,255,0.035)",
    font: { family: "JetBrains Mono, monospace", color: "#71717a", size: 10 },
};

function baseLayout(height = 400, extra = {}) {
    const base = {
        paper_bgcolor: C.bg,
        plot_bgcolor: C.plotBg,
        height,
        margin: { l: 50, r: 16, t: 12, b: 28 },
        font: C.font,
        showlegend: true,
        legend: {
            orientation: "h",
            yanchor: "bottom",
            y: 1.02,
            xanchor: "center",
            x: 0.5,
            bgcolor: "rgba(0,0,0,0)",
            font: { size: 10 },
        },
        xaxis: { gridcolor: C.grid, zeroline: false },
        yaxis: { gridcolor: C.grid, zeroline: false },
    };
    return { ...base, ...extra };
}

const PLOTLY_CFG = { responsive: true, displayModeBar: false };

// ═══════════════════════════════════════════════════════════════════
// NAVIGATION
// ═══════════════════════════════════════════════════════════════════

let currentPage = "overview";
const pageCache = {};

function navigate(page) {
    currentPage = page;
    $$(".nav-item").forEach((n) =>
        n.classList.toggle("active", n.dataset.page === page)
    );
    $$(".page").forEach((p) =>
        p.classList.toggle("active", p.id === `page-${page}`)
    );
    renderPage(page);
}

async function renderPage(page) {
    const el = $(`#page-${page}`);
    if (!el) return;
    el.innerHTML = loading();
    try {
        switch (page) {
            case "overview":
                await renderOverview(el);
                break;
            case "analysis":
                await renderAnalysis(el);
                break;
            case "signals":
                await renderSignals(el);
                break;
            case "portfolio":
                await renderPortfolio(el);
                break;
            case "agents":
                await renderAgents(el);
                break;
            case "settings":
                renderSettings(el);
                break;
        }
    } catch (err) {
        console.error(`Page ${page} error:`, err);
        el.innerHTML =
            pageHeader(page.charAt(0).toUpperCase() + page.slice(1), "") +
            emptyState(`Failed to load ${page}`, err.message);
    }
}

// ═══════════════════════════════════════════════════════════════════
// PAGE: OVERVIEW
// ═══════════════════════════════════════════════════════════════════

async function renderOverview(el) {
    const [indices, sectors] = await Promise.all([
        api("/api/indices"),
        api("/api/sectors"),
    ]);

    let html = pageHeader("Overview", "Market data & portfolio snapshot", [
        { type: "live", label: "Live", dot: true },
        { type: "ai", label: "Real-Time Data" },
    ]);

    // Index strip
    html += metricGrid(
        indices.map((i) => ({
            label: i.name,
            value: fmtPrice(i.price),
            delta: fmtPct(i.change_pct),
        }))
    );

    // Main grid
    html += '<div class="grid-2-1">';

    // S&P 500 chart
    html += `
        <div class="card">
            <div class="card-header">
                <span class="card-title">S&P 500</span>
                <span class="badge badge-live" style="font-size:10px;">6M</span>
            </div>
            <div class="card-body flush chart-wrap" id="ov-chart" style="min-height:320px;">
                ${loading()}
            </div>
        </div>`;

    // Sectors
    html += `
        <div class="card">
            <div class="card-header">
                <span class="card-title">Sector Performance</span>
            </div>
            <div class="card-body flush">
                ${sectors
                    .sort((a, b) => (b.change_pct || 0) - (a.change_pct || 0))
                    .map((s) => {
                        const pct = s.change_pct || 0;
                        const cls = pct >= 0 ? "val-up" : "val-down";
                        return `<div class="data-row">
                            <span class="data-sym">${s.sector}</span>
                            <span class="data-val ${cls}">${fmtPct(pct)}</span>
                        </div>`;
                    })
                    .join("")}
            </div>
        </div>`;

    html += "</div>"; // end grid

    el.innerHTML = html;

    // Load S&P chart async
    try {
        const sp = await api("/api/stock/^GSPC?period=6mo");
        if (sp.date && sp.close) {
            const pct =
                sp.close.length > 1
                    ? (sp.close[sp.close.length - 1] / sp.close[0] - 1) * 100
                    : 0;
            const col = pct >= 0 ? C.green : C.red;
            const fill =
                pct >= 0
                    ? "rgba(16,185,129,0.07)"
                    : "rgba(244,63,94,0.07)";

            Plotly.newPlot(
                "ov-chart",
                [
                    {
                        x: sp.date,
                        y: sp.close,
                        type: "scatter",
                        fill: "tozeroy",
                        fillcolor: fill,
                        line: { color: col, width: 2.5 },
                        hovertemplate: "%{x}<br>$%{y:,.2f}<extra></extra>",
                    },
                ],
                baseLayout(320, {
                    showlegend: false,
                    xaxis: { showgrid: false },
                    margin: { l: 52, r: 16, t: 8, b: 28 },
                }),
                PLOTLY_CFG
            );
        }
    } catch (e) {
        const c = $("#ov-chart");
        if (c) c.innerHTML = emptyState("Chart unavailable");
    }
}

// ═══════════════════════════════════════════════════════════════════
// PAGE: ANALYSIS
// ═══════════════════════════════════════════════════════════════════

let analysisState = { ticker: "AAPL", period: "1y" };

async function renderAnalysis(el) {
    el.innerHTML =
        pageHeader("Technical Analysis", "Multi-timeframe charting with indicators") +
        `<div class="input-row">
            <div class="input-group">
                <label class="input-label">Symbol</label>
                <input class="input" id="a-ticker" value="${analysisState.ticker}" style="width:140px" />
            </div>
            <div class="input-group">
                <label class="input-label">Period</label>
                <select class="select" id="a-period">
                    <option value="1mo">1 Month</option>
                    <option value="3mo">3 Months</option>
                    <option value="6mo">6 Months</option>
                    <option value="1y">1 Year</option>
                    <option value="2y">2 Years</option>
                </select>
            </div>
            <button class="btn btn-primary" id="a-go">Analyze</button>
        </div>
        <div id="a-metrics"></div>
        <div class="card">
            <div class="card-body flush chart-wrap" id="a-chart" style="min-height:400px;">
                ${loading()}
            </div>
        </div>
        <div id="a-summary"></div>`;

    const sel = $("#a-period");
    if (sel) sel.value = analysisState.period;
    $("#a-go").addEventListener("click", loadAnalysis);
    $("#a-ticker").addEventListener("keydown", (e) => {
        if (e.key === "Enter") loadAnalysis();
    });
    await loadAnalysis();
}

async function loadAnalysis() {
    const ticker = ($("#a-ticker")?.value || "AAPL").toUpperCase();
    const period = $("#a-period")?.value || "1y";
    analysisState = { ticker, period };

    const chart = $("#a-chart");
    const met = $("#a-metrics");
    const sum = $("#a-summary");
    if (chart) chart.innerHTML = loading();

    try {
        const [data, quote] = await Promise.all([
            api(`/api/indicators/${ticker}?period=${period}`),
            api(`/api/quote/${ticker}`),
        ]);

        if (data.error) {
            if (chart) chart.innerHTML = emptyState(`No data for ${ticker}`);
            return;
        }

        // Metrics
        if (met) {
            met.innerHTML = metricGrid([
                {
                    label: "Price",
                    value: `$${fmtPrice(quote.price)}`,
                    delta: fmtPct(quote.change_pct),
                },
                { label: "Open", value: `$${fmtPrice(quote.open)}` },
                { label: "Day High", value: `$${fmtPrice(quote.high)}` },
                { label: "Day Low", value: `$${fmtPrice(quote.low)}` },
                { label: "Volume", value: fmtNum(quote.volume) },
                { label: "Mkt Cap", value: fmtNum(quote.market_cap, "$") },
            ]);
        }

        // Chart
        renderMultiChart(data, "a-chart");

        // Summary
        if (sum && data.close?.length) {
            const last = data.close[data.close.length - 1];
            const sma20 = data.sma_20?.[data.sma_20.length - 1];
            const sma50 = data.sma_50?.[data.sma_50.length - 1];
            const rsi = data.rsi?.[data.rsi.length - 1];
            const macd = data.macd?.[data.macd.length - 1];
            const sig = data.macd_signal?.[data.macd_signal.length - 1];

            const items = [];
            if (sma20 != null) {
                const above = last > sma20;
                items.push({
                    label: "SMA 20",
                    value: above ? "Above" : "Below",
                    color: above ? "var(--green)" : "var(--red)",
                });
            }
            if (sma50 != null) {
                const above = last > sma50;
                items.push({
                    label: "SMA 50",
                    value: above ? "Above" : "Below",
                    color: above ? "var(--green)" : "var(--red)",
                });
            }
            if (rsi != null) {
                const label =
                    rsi > 70
                        ? "Overbought"
                        : rsi < 30
                        ? "Oversold"
                        : "Neutral";
                const color =
                    rsi > 70
                        ? "var(--red)"
                        : rsi < 30
                        ? "var(--green)"
                        : "var(--text-2)";
                items.push({
                    label: `RSI (${rsi.toFixed(1)})`,
                    value: label,
                    color,
                });
            }
            if (macd != null && sig != null) {
                const bull = macd > sig;
                items.push({
                    label: "MACD",
                    value: bull ? "Bullish" : "Bearish",
                    color: bull ? "var(--green)" : "var(--red)",
                });
            }

            sum.innerHTML = `
                <div class="divider"></div>
                <div class="section-title">Technical Summary</div>
                <div class="metric-grid" style="grid-template-columns:repeat(${items.length},1fr);">
                    ${items
                        .map(
                            (i) => `
                        <div class="metric-cell">
                            <div class="metric-label">${i.label}</div>
                            <div class="metric-value" style="font-size:16px;color:${i.color}">${i.value}</div>
                        </div>`
                        )
                        .join("")}
                </div>`;
        }
    } catch (err) {
        if (chart)
            chart.innerHTML = emptyState("Failed to load data", err.message);
    }
}

function renderMultiChart(data, containerId) {
    if (!data.date || !data.close) return;

    const hasRSI = data.rsi && data.rsi.some((v) => v != null);
    const hasMACD = data.macd && data.macd.some((v) => v != null);

    // Compute domains
    let priceDomain, volDomain, rsiDomain, macdDomain;
    if (hasRSI && hasMACD) {
        priceDomain = [0.42, 1];
        volDomain = [0.30, 0.39];
        rsiDomain = [0.15, 0.27];
        macdDomain = [0, 0.12];
    } else if (hasRSI) {
        priceDomain = [0.35, 1];
        volDomain = [0.2, 0.32];
        rsiDomain = [0, 0.17];
        macdDomain = null;
    } else {
        priceDomain = [0.25, 1];
        volDomain = [0, 0.22];
        rsiDomain = null;
        macdDomain = null;
    }

    const traces = [];

    // Candlestick
    traces.push({
        type: "candlestick",
        x: data.date,
        open: data.open,
        high: data.high,
        low: data.low,
        close: data.close,
        increasing: { line: { color: C.green }, fillcolor: C.green },
        decreasing: { line: { color: C.red }, fillcolor: C.red },
        name: "OHLC",
        yaxis: "y",
        showlegend: false,
    });

    // Overlays
    if (data.sma_20) {
        traces.push({
            x: data.date,
            y: data.sma_20,
            line: { color: C.indigo, width: 1.2 },
            name: "SMA 20",
            yaxis: "y",
        });
    }
    if (data.sma_50) {
        traces.push({
            x: data.date,
            y: data.sma_50,
            line: { color: C.sky, width: 1.2 },
            name: "SMA 50",
            yaxis: "y",
        });
    }
    if (data.bb_upper && data.bb_lower) {
        traces.push({
            x: data.date,
            y: data.bb_upper,
            line: { color: "rgba(99,102,241,0.25)", width: 1, dash: "dash" },
            name: "BB",
            yaxis: "y",
            showlegend: true,
        });
        traces.push({
            x: data.date,
            y: data.bb_lower,
            line: { color: "rgba(99,102,241,0.25)", width: 1, dash: "dash" },
            fill: "tonexty",
            fillcolor: "rgba(99,102,241,0.03)",
            name: "BB Low",
            yaxis: "y",
            showlegend: false,
        });
    }

    // Volume
    if (data.volume) {
        traces.push({
            type: "bar",
            x: data.date,
            y: data.volume,
            marker: {
                color: data.close.map((c, i) =>
                    c >= (data.open[i] || c) ? C.green : C.red
                ),
                opacity: 0.45,
            },
            name: "Volume",
            yaxis: "y2",
            showlegend: false,
        });
    }

    // RSI
    if (hasRSI) {
        traces.push({
            x: data.date,
            y: data.rsi,
            line: { color: C.indigo, width: 1.5 },
            name: "RSI",
            yaxis: "y3",
        });
    }

    // MACD
    if (hasMACD) {
        traces.push({
            type: "bar",
            x: data.date,
            y: data.macd_hist,
            marker: {
                color: (data.macd_hist || []).map((v) =>
                    (v || 0) >= 0 ? C.green : C.red
                ),
            },
            name: "Hist",
            yaxis: "y4",
            showlegend: false,
        });
        traces.push({
            x: data.date,
            y: data.macd,
            line: { color: C.sky, width: 1 },
            name: "MACD",
            yaxis: "y4",
        });
        traces.push({
            x: data.date,
            y: data.macd_signal,
            line: { color: C.indigo, width: 1 },
            name: "Signal",
            yaxis: "y4",
        });
    }

    // Shapes (RSI reference lines)
    const shapes = [];
    if (hasRSI && data.date.length > 1) {
        const x0 = data.date[0];
        const x1 = data.date[data.date.length - 1];
        shapes.push(
            {
                type: "line", xref: "x", yref: "y3",
                x0, x1, y0: 70, y1: 70,
                line: { color: "rgba(244,63,94,0.4)", width: 1, dash: "dot" },
            },
            {
                type: "line", xref: "x", yref: "y3",
                x0, x1, y0: 30, y1: 30,
                line: { color: "rgba(16,185,129,0.4)", width: 1, dash: "dot" },
            }
        );
    }

    const layoutObj = {
        paper_bgcolor: C.bg,
        plot_bgcolor: C.plotBg,
        height: 660,
        margin: { l: 52, r: 16, t: 8, b: 28 },
        font: C.font,
        showlegend: true,
        legend: {
            orientation: "h",
            y: 1.03,
            x: 0.5,
            xanchor: "center",
            bgcolor: "rgba(0,0,0,0)",
            font: { size: 10 },
        },
        xaxis: {
            rangeslider: { visible: false },
            gridcolor: C.grid,
            zeroline: false,
        },
        yaxis: {
            title: { text: "Price", font: { size: 10 } },
            gridcolor: C.grid,
            domain: priceDomain,
            zeroline: false,
        },
        yaxis2: {
            title: { text: "Vol", font: { size: 10 } },
            gridcolor: C.grid,
            domain: volDomain,
            zeroline: false,
        },
        shapes,
    };

    if (hasRSI && rsiDomain) {
        layoutObj.yaxis3 = {
            title: { text: "RSI", font: { size: 10 } },
            gridcolor: C.grid,
            domain: rsiDomain,
            zeroline: false,
        };
    }
    if (hasMACD && macdDomain) {
        layoutObj.yaxis4 = {
            title: { text: "MACD", font: { size: 10 } },
            gridcolor: C.grid,
            domain: macdDomain,
            zeroline: false,
        };
    }

    Plotly.newPlot(containerId, traces, layoutObj, PLOTLY_CFG);
}

// ═══════════════════════════════════════════════════════════════════
// PAGE: SIGNALS
// ═══════════════════════════════════════════════════════════════════

async function renderSignals(el) {
    el.innerHTML =
        pageHeader("Alpha Signals", "AI-powered factor decomposition", [
            { type: "ai", label: "ML Engine", dot: true, dotClass: "open" },
        ]) + loading();

    try {
        const result = await api("/api/signals");
        const signals = result.data || [];

        if (!signals.length) {
            el.innerHTML =
                pageHeader("Alpha Signals", "AI-powered factor decomposition") +
                emptyState(
                    "No alpha signals available",
                    "python download_data.py"
                );
            return;
        }

        const strongBuy = signals.filter(
            (s) => (s.composite_alpha || 0) > 0.5
        ).length;
        const strongSell = signals.filter(
            (s) => (s.composite_alpha || 0) < -0.5
        ).length;
        const avgConf =
            signals.reduce((a, s) => a + (s.confidence || 0), 0) /
            signals.length;

        let html = pageHeader(
            "Alpha Signals",
            "AI-powered factor decomposition",
            [{ type: "ai", label: "ML Engine", dot: true, dotClass: "open" }]
        );

        html += metricGrid([
            { label: "Total Signals", value: String(signals.length) },
            { label: "Strong Buy", value: String(strongBuy) },
            { label: "Strong Sell", value: String(strongSell) },
            { label: "Avg Confidence", value: `${(avgConf * 100).toFixed(1)}%` },
        ]);

        // Tabs
        html += `<div class="tab-bar" id="sig-tabs">
            <div class="tab active" data-tab="long">Long Ideas</div>
            <div class="tab" data-tab="short">Short Ideas</div>
            <div class="tab" data-tab="all">Full Matrix</div>
        </div>`;

        // Compute sorted views
        const sorted = [...signals].sort(
            (a, b) => (b.composite_alpha || 0) - (a.composite_alpha || 0)
        );
        const longs = sorted.slice(0, 15);
        const shorts = sorted.slice(-15).reverse();

        function renderTable(items) {
            return `<div class="card"><div class="card-body flush">
                <table class="data-table">
                    <thead><tr>
                        <th>Symbol</th><th>Alpha</th><th>Confidence</th><th>Rank</th>
                    </tr></thead>
                    <tbody>${items
                        .map((s) => {
                            const alpha = s.composite_alpha || 0;
                            const cls =
                                alpha > 0.2
                                    ? "text-green"
                                    : alpha < -0.2
                                    ? "text-red"
                                    : "text-muted";
                            return `<tr>
                                <td style="color:var(--text-1);font-weight:600">${
                                    s.symbol || "—"
                                }</td>
                                <td class="${cls}">${alpha.toFixed(4)}</td>
                                <td>${((s.confidence || 0) * 100).toFixed(1)}%</td>
                                <td>${s.alpha_rank ?? "—"}</td>
                            </tr>`;
                        })
                        .join("")}
                    </tbody>
                </table>
            </div></div>`;
        }

        html += `<div id="sig-long">${renderTable(longs)}</div>`;
        html += `<div id="sig-short" style="display:none">${renderTable(
            shorts
        )}</div>`;
        html += `<div id="sig-all" style="display:none">${renderTable(
            sorted
        )}</div>`;

        el.innerHTML = html;

        // Tab switching
        $$("#sig-tabs .tab").forEach((tab) => {
            tab.addEventListener("click", () => {
                $$("#sig-tabs .tab").forEach((t) =>
                    t.classList.remove("active")
                );
                tab.classList.add("active");
                const t = tab.dataset.tab;
                ["long", "short", "all"].forEach((id) => {
                    const panel = $(`#sig-${id}`);
                    if (panel) panel.style.display = id === t ? "" : "none";
                });
            });
        });
    } catch (err) {
        el.innerHTML =
            pageHeader("Alpha Signals", "") +
            emptyState("Failed to load signals", err.message);
    }
}

// ═══════════════════════════════════════════════════════════════════
// PAGE: PORTFOLIO
// ═══════════════════════════════════════════════════════════════════

const DEFAULT_PORTFOLIO = {
    AAPL: { shares: 100, cost: 175.0 },
    MSFT: { shares: 50, cost: 380.0 },
    NVDA: { shares: 30, cost: 450.0 },
    GOOGL: { shares: 25, cost: 140.0 },
    AMZN: { shares: 40, cost: 178.0 },
};

async function renderPortfolio(el) {
    el.innerHTML =
        pageHeader("Portfolio", "Position tracking & risk attribution") +
        loading();

    try {
        const quotes = await Promise.all(
            Object.keys(DEFAULT_PORTFOLIO).map((sym) =>
                api(`/api/quote/${sym}`).then((q) => ({ sym, ...q }))
            )
        );

        let totalValue = 0,
            totalCost = 0;
        const holdings = [];

        for (const q of quotes) {
            const pos = DEFAULT_PORTFOLIO[q.sym];
            const price = q.price || pos.cost;
            const value = price * pos.shares;
            const cost = pos.cost * pos.shares;
            const pnl = value - cost;
            const pnlPct = cost > 0 ? (pnl / cost) * 100 : 0;
            totalValue += value;
            totalCost += cost;
            holdings.push({
                sym: q.sym,
                name: q.name || q.sym,
                shares: pos.shares,
                cost: pos.cost,
                price,
                value,
                pnl,
                pnlPct,
            });
        }

        const totalPnl = totalValue - totalCost;
        const totalPnlPct = totalCost > 0 ? (totalPnl / totalCost) * 100 : 0;

        let html = pageHeader(
            "Portfolio",
            "Position tracking & risk attribution"
        );

        html += metricGrid([
            { label: "Portfolio Value", value: `$${fmtPrice(totalValue)}` },
            { label: "Total Cost", value: `$${fmtPrice(totalCost)}` },
            {
                label: "Total P&L",
                value: `$${fmtPrice(totalPnl)}`,
                delta: fmtPct(totalPnlPct),
            },
            { label: "Positions", value: String(holdings.length) },
        ]);

        html += '<div class="grid-2-1">';

        // Holdings table
        html += `<div class="card">
            <div class="card-header"><span class="card-title">Holdings</span></div>
            <div class="card-body flush">
                <table class="data-table">
                    <thead><tr>
                        <th>Symbol</th><th>Shares</th><th>Cost</th><th>Price</th><th>Value</th><th>P&L</th><th>Return</th>
                    </tr></thead>
                    <tbody>${holdings
                        .map((h) => {
                            const cls =
                                h.pnl >= 0 ? "text-green" : "text-red";
                            return `<tr>
                                <td style="color:var(--text-1);font-weight:600">${h.sym}</td>
                                <td>${h.shares}</td>
                                <td>$${h.cost.toFixed(2)}</td>
                                <td>$${h.price.toFixed(2)}</td>
                                <td>$${fmtPrice(h.value)}</td>
                                <td class="${cls}">$${fmtPrice(h.pnl)}</td>
                                <td class="${cls}">${fmtPct(h.pnlPct)}</td>
                            </tr>`;
                        })
                        .join("")}
                    </tbody>
                </table>
            </div>
        </div>`;

        // Allocation pie
        html += `<div class="card">
            <div class="card-header"><span class="card-title">Allocation</span></div>
            <div class="card-body flush chart-wrap" id="port-pie" style="min-height:300px;"></div>
        </div>`;

        html += "</div>";
        el.innerHTML = html;

        // Render pie
        Plotly.newPlot(
            "port-pie",
            [
                {
                    type: "pie",
                    labels: holdings.map((h) => h.sym),
                    values: holdings.map((h) => h.value),
                    hole: 0.6,
                    marker: {
                        colors: [
                            C.indigo,
                            C.sky,
                            C.green,
                            C.amber,
                            C.red,
                            "#a78bfa",
                            "#f472b6",
                        ],
                    },
                    textfont: {
                        family: "JetBrains Mono",
                        size: 11,
                        color: "#a1a1aa",
                    },
                    hovertemplate:
                        "%{label}<br>$%{value:,.0f}<br>%{percent}<extra></extra>",
                },
            ],
            baseLayout(300, {
                margin: { l: 16, r: 16, t: 16, b: 16 },
                showlegend: true,
                legend: {
                    font: { size: 11 },
                    orientation: "h",
                    y: -0.05,
                    x: 0.5,
                    xanchor: "center",
                },
            }),
            PLOTLY_CFG
        );
    } catch (err) {
        el.innerHTML =
            pageHeader("Portfolio", "") +
            emptyState("Failed to load portfolio", err.message);
    }
}

// ═══════════════════════════════════════════════════════════════════
// PAGE: AGENTS
// ═══════════════════════════════════════════════════════════════════

const PIPELINE_STAGES = [
    ["Data Ingest", "DataIngestionAgent", "01"],
    ["Quality", "DataQualityAgent", "02"],
    ["Features", "FeatureEngineeringAgent", "03"],
    ["Regime", "RegimeDetectionAgent", "04"],
    ["Model", "ModelingAgent", "05"],
    ["Decision", "DecisionAgent", "06"],
    ["Risk", "RiskAgent", "07"],
    ["Scenario", "ScenarioAgent", "08"],
    ["Monitor", "MonitoringAgent", "09"],
    ["Lifecycle", "LifecycleAgent", "10"],
];

async function renderAgents(el) {
    el.innerHTML =
        pageHeader(
            "Agent Pipeline",
            "Cooperative multi-agent execution engine",
            [
                {
                    type: "ai",
                    label: "Reasoning Engine",
                    dot: true,
                    dotClass: "open",
                },
            ]
        ) + loading();

    try {
        const status = await api("/api/agents/status");

        if (status.error && !status.agents?.length) {
            el.innerHTML =
                pageHeader(
                    "Agent Pipeline",
                    "Cooperative multi-agent execution engine"
                ) +
                `<div class="banner banner-error">${status.error}</div>`;
            return;
        }

        let html = pageHeader(
            "Agent Pipeline",
            "Cooperative multi-agent execution engine",
            [
                {
                    type: "ai",
                    label: "Reasoning Engine",
                    dot: true,
                    dotClass: "open",
                },
            ]
        );

        // Agent chips
        html += '<div class="section-title">Pipeline Status</div>';
        html += '<div class="grid-5">';
        for (const [label, agentName, num] of PIPELINE_STAGES) {
            const agent = (status.agents || []).find(
                (a) => a.name === agentName
            );
            const st = agent?.status || "missing";
            const chipCls =
                st === "healthy"
                    ? "ok"
                    : st === "degraded"
                    ? "warn"
                    : "err";
            const colorMap = {
                healthy: "var(--green)",
                degraded: "var(--amber)",
                error: "var(--red)",
                missing: "var(--red)",
                unknown: "var(--text-4)",
            };
            const col = colorMap[st] || colorMap.unknown;
            html += `
                <div class="agent-chip ${chipCls}">
                    <div class="agent-num">${num}</div>
                    <div class="agent-name">${label}</div>
                    <div class="agent-status-text" style="color:${col}">${st}</div>
                </div>`;
        }
        html += "</div>";

        // System metrics
        html += metricGrid([
            {
                label: "System",
                value: (status.system_status || "unknown").toUpperCase(),
            },
            {
                label: "Agents",
                value: String(status.agent_count || 0),
            },
            {
                label: "Published",
                value: String(status.bus_stats?.total_published || 0),
            },
            {
                label: "Consumed",
                value: String(status.bus_stats?.total_consumed || 0),
            },
        ]);

        // Run pipeline section
        html += '<div class="divider"></div>';
        html += '<div class="section-title">Run Pipeline</div>';
        html += `<div class="input-row">
            <div class="input-group">
                <label class="input-label">Symbols</label>
                <input class="input" id="ag-symbols" value="AAPL, MSFT, GOOGL" style="width:240px" />
            </div>
            <div class="input-group">
                <label class="input-label">Source</label>
                <select class="select" id="ag-source">
                    <option value="yahoo">Yahoo Finance</option>
                    <option value="openbb">OpenBB</option>
                </select>
            </div>
            <div class="input-group">
                <label class="input-label">Period</label>
                <select class="select" id="ag-period">
                    <option value="6mo">6 Months</option>
                    <option value="1y" selected>1 Year</option>
                    <option value="2y">2 Years</option>
                </select>
            </div>
            <button class="btn btn-primary" id="ag-run">Run Pipeline</button>
        </div>`;
        html += '<div id="ag-results"></div>';

        // Agent inspector
        html += '<div class="divider"></div>';
        html += '<div class="section-title">Agent Inspector</div>';
        const agentNames = (status.agents || []).map((a) => a.name);
        html += `<div class="input-row">
            <div class="input-group">
                <label class="input-label">Select Agent</label>
                <select class="select" id="ag-inspect">
                    ${agentNames
                        .map((n) => `<option value="${n}">${n}</option>`)
                        .join("")}
                </select>
            </div>
        </div>`;
        html += '<div id="ag-detail"></div>';

        // Bus stats
        html += '<div class="divider"></div>';
        html += '<div class="section-title">Event Bus</div>';
        html += `<div class="card"><div class="card-body">
            <pre class="font-mono text-sm" style="color:var(--text-2);white-space:pre-wrap;">${JSON.stringify(
                status.bus_stats || {},
                null,
                2
            )}</pre>
        </div></div>`;

        el.innerHTML = html;

        // Wire run button
        $("#ag-run")?.addEventListener("click", async () => {
            const symbols = ($("#ag-symbols")?.value || "AAPL")
                .split(",")
                .map((s) => s.trim())
                .filter(Boolean);
            const source = $("#ag-source")?.value || "yahoo";
            const period = $("#ag-period")?.value || "1y";
            const resultsEl = $("#ag-results");
            if (!resultsEl) return;

            resultsEl.innerHTML = loading();
            try {
                const results = await apiPost("/api/agents/run", {
                    symbols,
                    source,
                    period,
                });
                if (results.error) {
                    resultsEl.innerHTML = `<div class="banner banner-error">${results.error}</div>`;
                    return;
                }

                let rHtml = "";
                for (const [sym, res] of Object.entries(results)) {
                    const ok = res.success;
                    const icon = ok ? "✓" : "✗";
                    const iconCol = ok ? "var(--green)" : "var(--red)";
                    const dur = (res.duration_ms || 0).toFixed(0);

                    rHtml += `<div class="pipeline-result">
                        <div class="pipeline-result-header" onclick="this.parentElement.classList.toggle('open')">
                            <span style="color:${iconCol};font-weight:700;font-family:var(--font-mono)">${icon}</span>
                            <span style="font-weight:600;color:var(--text-1)">${sym}</span>
                            <span class="text-muted text-xs font-mono" style="margin-left:auto">${dur}ms</span>
                        </div>
                        <div class="pipeline-result-body">
                            <div style="margin-top:8px;">
                                ${PIPELINE_STAGES.map(([label, , num]) => {
                                    const completed =
                                        res.stages_completed || [];
                                    const done = completed.some(
                                        (s) =>
                                            s.toLowerCase().replace(/\s/g, "").includes(
                                                label.toLowerCase().replace(/\s/g, "")
                                            )
                                    );
                                    const col = done
                                        ? "var(--green)"
                                        : "var(--text-5)";
                                    const mark = done ? "done" : "skip";
                                    return `<span class="font-mono text-xs" style="color:${col};margin-right:12px;">[${mark}] ${num} ${label}</span>`;
                                }).join("")}
                            </div>
                            ${
                                res.data?.signal
                                    ? `<div class="metric-grid mt-4" style="grid-template-columns:repeat(4,1fr)">
                                        <div class="metric-cell"><div class="metric-label">Signal</div><div class="metric-value" style="font-size:15px">${
                                            res.data.signal
                                        }</div></div>
                                        <div class="metric-cell"><div class="metric-label">Confidence</div><div class="metric-value" style="font-size:15px">${
                                            res.data.confidence != null
                                                ? (
                                                      res.data.confidence * 100
                                                  ).toFixed(1) + "%"
                                                : "—"
                                        }</div></div>
                                        <div class="metric-cell"><div class="metric-label">Regime</div><div class="metric-value" style="font-size:15px">${
                                            res.data.regime || "—"
                                        }</div></div>
                                        <div class="metric-cell"><div class="metric-label">Conviction</div><div class="metric-value" style="font-size:15px">${
                                            res.data.conviction != null
                                                ? (
                                                      res.data.conviction * 100
                                                  ).toFixed(1) + "%"
                                                : "—"
                                        }</div></div>
                                    </div>`
                                    : ""
                            }
                        </div>
                    </div>`;
                }
                resultsEl.innerHTML = rHtml;
            } catch (err) {
                resultsEl.innerHTML = `<div class="banner banner-error">${err.message}</div>`;
            }
        });

        // Wire inspector
        const showAgent = () => {
            const name = $("#ag-inspect")?.value;
            const agent = (status.agents || []).find((a) => a.name === name);
            const detail = $("#ag-detail");
            if (!detail || !agent) return;

            detail.innerHTML = `<div class="grid-2">
                <div class="card">
                    <div class="card-header"><span class="card-title">Health</span></div>
                    <div class="card-body">
                        <pre class="font-mono text-sm" style="color:var(--text-2);white-space:pre-wrap;">${JSON.stringify(
                            agent.health || {},
                            null,
                            2
                        )}</pre>
                    </div>
                </div>
                <div class="card">
                    <div class="card-header"><span class="card-title">Metrics</span></div>
                    <div class="card-body">
                        <pre class="font-mono text-sm" style="color:var(--text-2);white-space:pre-wrap;">${JSON.stringify(
                            agent.metrics || {},
                            null,
                            2
                        )}</pre>
                    </div>
                </div>
            </div>`;
        };

        $("#ag-inspect")?.addEventListener("change", showAgent);
        showAgent();
    } catch (err) {
        el.innerHTML =
            pageHeader("Agent Pipeline", "") +
            emptyState("Agent system unavailable", err.message);
    }
}

// ═══════════════════════════════════════════════════════════════════
// PAGE: SETTINGS
// ═══════════════════════════════════════════════════════════════════

function renderSettings(el) {
    const info = [
        ["Framework", "FastAPI + Vanilla JS"],
        ["Design System", "Midnight SaaS"],
        ["Chart Engine", "Plotly.js"],
        ["Agent Engine", "Aegis Orchestrator"],
        ["Data Source", "Yahoo Finance"],
        ["Version", "2.0"],
    ];

    el.innerHTML =
        pageHeader("Settings", "Configuration & system info") +
        `<div class="section-title">Actions</div>
        <div style="display:flex;gap:12px;margin-bottom:24px;">
            <button class="btn btn-ghost" onclick="location.reload()">Reload Dashboard</button>
            <button class="btn btn-ghost" onclick="window.open('/docs','_blank')">API Docs</button>
        </div>
        <div class="divider"></div>
        <div class="section-title">System Information</div>
        <div class="card">
            <div class="card-body flush">
                ${info
                    .map(
                        ([k, v]) => `
                    <div class="data-row">
                        <span style="font-size:13px;color:var(--text-3)">${k}</span>
                        <span class="font-mono" style="font-size:13px;color:var(--text-1)">${v}</span>
                    </div>`
                    )
                    .join("")}
            </div>
        </div>`;
}

// ═══════════════════════════════════════════════════════════════════
// QUICK QUOTE (Sidebar)
// ═══════════════════════════════════════════════════════════════════

let quoteTimeout = null;

function setupQuickQuote() {
    const input = $("#quick-ticker");
    if (!input) return;

    const load = async () => {
        const sym = input.value.trim().toUpperCase();
        const el = $("#quick-quote-result");
        if (!sym || !el) return;

        el.innerHTML =
            '<div class="loading-wrap" style="padding:8px 0;"><div class="spinner"></div></div>';

        try {
            const q = await api(`/api/quote/${sym}`);
            if (!q.price) {
                el.innerHTML = "";
                return;
            }
            const cls = q.change >= 0 ? "up" : "down";
            const sign = q.change >= 0 ? "+" : "";
            el.innerHTML = `
                <div class="quote-card">
                    <div class="quote-name">${q.name}</div>
                    <div class="quote-price">$${q.price.toFixed(2)}</div>
                    <div class="quote-delta ${cls}">${sign}${q.change.toFixed(
                2
            )} (${sign}${q.change_pct.toFixed(2)}%)</div>
                </div>`;
        } catch {
            el.innerHTML = "";
        }
    };

    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            clearTimeout(quoteTimeout);
            load();
        }
    });

    input.addEventListener("input", () => {
        clearTimeout(quoteTimeout);
        quoteTimeout = setTimeout(load, 800);
    });

    // Initial load
    load();
}

// ═══════════════════════════════════════════════════════════════════
// MARKET STATUS
// ═══════════════════════════════════════════════════════════════════

function updateMarketStatus() {
    const el = $("#market-status");
    if (!el) return;
    const now = new Date();
    const h = now.getHours();
    const d = now.getDay();
    const open = d >= 1 && d <= 5 && h >= 9 && h < 16;
    const cls = open ? "open" : "closed";
    const text = open ? "Market Open" : "Market Closed";
    const time = now.toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
    });
    el.innerHTML = `<span class="status-dot ${cls}"></span> ${text} · ${time}`;
}

// ═══════════════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════════════

function init() {
    // Navigation
    $$(".nav-item").forEach((item) => {
        item.addEventListener("click", (e) => {
            e.preventDefault();
            navigate(item.dataset.page);
        });
    });

    // Quick quote
    setupQuickQuote();

    // Market status
    updateMarketStatus();
    setInterval(updateMarketStatus, 30000);

    // Load initial page
    navigate("overview");
}

document.addEventListener("DOMContentLoaded", init);
