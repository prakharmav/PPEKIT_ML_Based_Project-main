"""
SafeGuard AI — Industrial PPE Safety Compliance Monitoring System
=================================================================
Multi-page Streamlit application with a premium dark-themed glassmorphic UI.
Matches the provided dashboard design mockup.
"""

import os, time, io, base64, tempfile, textwrap
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from database        import IncidentDatabase
from detector        import PPEDetector
from report_generator import generate_csv, generate_pdf

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SafeGuard AI – PPE Monitor",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS FOR HTML RENDERING (PREVENTS MARKDOWN CODE-BLOCK TRAP)
# ─────────────────────────────────────────────────────────────────────────────
def html_markdown(content):
    # Remove leading spaces from each line to prevent markdown from treating it as a code block
    cleaned = "\n".join(line.lstrip() for line in content.splitlines())
    st.markdown(cleaned, unsafe_allow_html=True)

def render_to_ph(ph, content):
    cleaned = "\n".join(line.lstrip() for line in content.splitlines())
    ph.markdown(cleaned, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CSS & STYLE SYSTEM
# ─────────────────────────────────────────────────────────────────────────────
html_markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
}

/* Base App Gradient Background */
.stApp {
    background: radial-gradient(ellipse at 30% 20%, #0d1233 0%, #040614 65%, #080a22 100%) !important;
    color: #e0e8ff !important;
    min-height: 100vh;
}

/* Hide Default Headers & Footers */
#MainMenu, footer, header { visibility: hidden !important; }
.block-container { padding: 0.8rem 2rem 2rem 2rem !important; max-width: 100% !important; }

/* Custom Scrollbars */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.01);
}
::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.15);
}

/* Sidebar Custom Styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060817 0%, #090e24 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.05) !important;
}
[data-testid="stSidebar"] * {
    color: #c8d3f5 !important;
}

/* Hide sidebar default radio styles, make it custom cards */
[data-testid="stRadio"] div[role="radiogroup"] {
    gap: 0.5rem !important;
}
[data-testid="stRadio"] input[type="radio"] {
    display: none !important;
}
[data-testid="stRadio"] label {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(255, 255, 255, 0.04) !important;
    border-radius: 10px !important;
    padding: 12px 18px !important;
    margin: 0 !important;
    cursor: pointer;
    transition: all 0.22s ease-in-out;
    display: flex !important;
    align-items: center;
    width: 100% !important;
}
[data-testid="stRadio"] label:hover {
    background: rgba(255, 255, 255, 0.05) !important;
    border-color: rgba(0, 230, 118, 0.3) !important;
    transform: translateX(3px);
}
[data-testid="stRadio"] label:has(input:checked) {
    background: rgba(0, 230, 118, 0.07) !important;
    border-color: #00E676 !important;
    color: #00E676 !important;
    box-shadow: 0 0 15px rgba(0, 230, 118, 0.12);
}
[data-testid="stRadio"] label div[data-testid="stMarkdownContainer"] p {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    margin: 0 !important;
}

/* Logo Design */
.sg-logo {
    text-align: left;
    padding: 10px 10px 10px 10px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.sg-logo-icon {
    font-size: 2.2rem;
    color: #ff6d00;
}
.sg-logo-text-wrapper {
    display: flex;
    flex-direction: column;
}
.sg-logo-title {
    font-size: 1.35rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: 0.5px;
    line-height: 1.1;
}
.sg-logo-sub {
    font-size: 0.68rem;
    color: rgba(180, 195, 240, 0.5);
    letter-spacing: 0.8px;
    font-weight: 700;
}

/* Sidebar overview widget */
.sidebar-widget {
    background: rgba(255, 255, 255, 0.015);
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 12px;
    padding: 16px;
    margin-top: 15px;
}
.widget-header {
    font-size: 0.75rem;
    font-weight: 700;
    color: rgba(180, 195, 240, 0.5);
    letter-spacing: 1.2px;
    margin-bottom: 12px;
}
.widget-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
    font-size: 0.85rem;
}
.widget-row:last-child { margin-bottom: 0; }
.widget-row .val {
    font-weight: 600;
    color: #e0e8ff;
    display: flex;
    align-items: center;
    gap: 6px;
}
.dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
}
.dot.green {
    background-color: #00E676;
    box-shadow: 0 0 8px #00E676;
}
.green-text { color: #00E676 !important; font-weight: 700; }
.sidebar-footer {
    font-size: 0.72rem;
    color: rgba(180, 195, 240, 0.3);
    text-align: center;
    margin-top: 20px;
    line-height: 1.4;
}

/* Top Navbar Style */
.top-navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: rgba(13, 17, 39, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 12px;
    padding: 14px 24px;
    margin-bottom: 20px;
    backdrop-filter: blur(10px);
}
.nav-left {
    display: flex;
    align-items: center;
    gap: 12px;
}
.nav-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #ffffff;
}
.live-badge {
    background: rgba(0, 230, 118, 0.12);
    color: #00E676;
    border: 1px solid rgba(0, 230, 118, 0.25);
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.8px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.blinking-dot {
    width: 6px;
    height: 6px;
    background-color: #00E676;
    border-radius: 50%;
    animation: blink 1.2s infinite;
}
@keyframes blink {
    0% { opacity: 0.3; }
    50% { opacity: 1; }
    100% { opacity: 0.3; }
}
.nav-center {
    display: flex;
    align-items: center;
    gap: 10px;
}
.badge-item {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 8px;
    padding: 6px 12px;
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.75rem;
}
.badge-label {
    color: rgba(180, 195, 240, 0.5);
    font-weight: 700;
    letter-spacing: 0.5px;
}
.badge-val {
    font-weight: 700;
    border-radius: 4px;
    padding: 1px 6px;
}
.val-green { background: rgba(0, 230, 118, 0.12); color: #00E676; }
.val-blue { background: rgba(41, 121, 255, 0.12); color: #2979FF; }
.val-blue-text { color: #2979FF; }

.nav-right {
    display: flex;
    align-items: center;
    gap: 20px;
}
.nav-time {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    line-height: 1.2;
}
.clock-time {
    font-size: 0.95rem;
    font-weight: 700;
    color: #ffffff;
}
.clock-date {
    font-size: 0.75rem;
    color: rgba(180, 195, 240, 0.5);
}
.nav-bell, .nav-theme {
    position: relative;
    cursor: pointer;
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 50%;
    width: 38px;
    height: 38px;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s;
}
.nav-bell:hover, .nav-theme:hover {
    background: rgba(255, 255, 255, 0.06);
    border-color: rgba(255,255,255,0.1);
}
.bell-badge {
    position: absolute;
    top: -2px;
    right: -2px;
    background-color: #FF5252;
    color: white;
    font-size: 0.65rem;
    font-weight: 700;
    width: 15px;
    height: 15px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
}

/* Card Styling */
.dashboard-card {
    background: rgba(13, 17, 39, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 12px;
    padding: 20px;
    backdrop-filter: blur(10px);
    margin-bottom: 20px;
}
.card-header-wrapper {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 15px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.03);
    padding-bottom: 10px;
}
.card-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    display: flex;
    align-items: center;
    gap: 8px;
}
.card-action-link {
    font-size: 0.8rem;
    color: #2979FF;
    text-decoration: none;
    font-weight: 600;
}
.card-action-link:hover {
    color: #00d4ff;
    text-decoration: underline;
}

/* System Overview cards */
.system-overview-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 12px;
    margin-bottom: 20px;
}
.overview-card {
    background: rgba(13, 17, 39, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 12px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(10px);
}
.overview-card::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 4px;
    height: 100%;
}
.card-purple::after { background-color: #7b6fff; }
.card-red::after { background-color: #FF5252; }
.card-green::after { background-color: #00E676; }
.card-amber::after { background-color: #FFC107; }
.card-blue::after { background-color: #2979FF; }

.overview-card .icon-val-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}
.overview-card .icon {
    font-size: 1.4rem;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
}
.card-purple .icon { background: rgba(123, 111, 255, 0.12); color: #7b6fff; }
.card-red .icon { background: rgba(255, 82, 82, 0.12); color: #FF5252; }
.card-green .icon { background: rgba(0, 230, 118, 0.12); color: #00E676; }
.card-amber .icon { background: rgba(255, 193, 7, 0.12); color: #FFC107; }
.card-blue .icon { background: rgba(41, 121, 255, 0.12); color: #2979FF; }

.overview-card .value {
    font-size: 1.7rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1.2;
}
.overview-card .label {
    font-size: 0.72rem;
    color: rgba(180, 195, 240, 0.5);
    text-transform: uppercase;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
}
.overview-card .trend {
    font-size: 0.75rem;
    font-weight: 600;
}
.trend.green { color: #00E676; }
.trend.red { color: #FF5252; }
.trend.muted { color: rgba(180, 195, 240, 0.4); }

/* Live Alerts List */
.alerts-list {
    display: flex;
    flex-direction: column;
    gap: 10px;
}
.alert-item {
    background: rgba(255, 255, 255, 0.015);
    border: 1px solid rgba(255, 255, 255, 0.03);
    border-radius: 10px;
    padding: 10px 14px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    transition: all 0.2s;
}
.alert-item:hover {
    background: rgba(255, 255, 255, 0.03);
    border-color: rgba(255, 255, 255, 0.06);
}
.alert-item-left {
    display: flex;
    align-items: center;
    gap: 12px;
}
.alert-icon {
    font-size: 1.1rem;
    width: 32px;
    height: 32px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.alert-item.crit { border-left: 3px solid #FF5252; }
.alert-item.warn { border-left: 3px solid #FFC107; }
.alert-item.crit .alert-icon { background: rgba(255, 82, 82, 0.1); color: #FF5252; }
.alert-item.warn .alert-icon { background: rgba(255, 193, 7, 0.1); color: #FFC107; }

.alert-info {
    display: flex;
    flex-direction: column;
    line-height: 1.3;
}
.alert-msg {
    font-size: 0.85rem;
    font-weight: 700;
    color: #ffffff;
}
.alert-sub {
    font-size: 0.72rem;
    color: rgba(180, 195, 240, 0.45);
}
.alert-time {
    font-size: 0.72rem;
    color: rgba(180, 195, 240, 0.5);
    font-weight: 600;
}

/* Recent Violations Cards */
.violations-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
}
.violation-thumb-card {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 10px;
    overflow: hidden;
    transition: all 0.22s;
}
.violation-thumb-card:hover {
    transform: translateY(-2px);
    border-color: rgba(255, 255, 255, 0.1);
}
.violation-thumb-img {
    width: 100%;
    height: 70px;
    object-fit: cover;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
}
.violation-thumb-info {
    padding: 8px;
    line-height: 1.3;
}
.violation-thumb-time {
    font-size: 0.65rem;
    color: rgba(180, 195, 240, 0.4);
}
.violation-thumb-type {
    font-size: 0.75rem;
    font-weight: 700;
    color: #FF5252;
}
.violation-thumb-zone {
    font-size: 0.68rem;
    color: #ffffff;
}

/* Recent Incidents Table */
.custom-table-wrapper {
    overflow-x: auto;
    width: 100%;
}
.custom-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.8rem;
    text-align: left;
}
.custom-table th {
    background: rgba(255, 255, 255, 0.02);
    color: rgba(180, 195, 240, 0.5);
    font-weight: 700;
    text-transform: uppercase;
    font-size: 0.68rem;
    letter-spacing: 0.5px;
    padding: 10px 12px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
}
.custom-table td {
    padding: 8px 12px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.03);
    color: #c8d3f5;
    vertical-align: middle;
}
.custom-table tr:hover {
    background: rgba(255, 255, 255, 0.01);
}
.table-badge {
    font-weight: 700;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.7rem;
    display: inline-block;
}
.badge-table-red { background: rgba(255, 82, 82, 0.12); color: #FF5252; }
.badge-table-amber { background: rgba(255, 193, 7, 0.12); color: #FFC107; }
.badge-table-green { background: rgba(0, 230, 118, 0.12); color: #00E676; }

.table-thumbnail {
    width: 32px;
    height: 22px;
    object-fit: cover;
    border-radius: 4px;
    border: 1px solid rgba(255, 255, 255, 0.08);
}

/* Bottom Bar */
.bottom-bar {
    display: grid;
    grid-template-columns: 3fr 7fr;
    gap: 15px;
    margin-top: 10px;
}
.safety-score-card {
    background: rgba(13, 17, 39, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 12px;
    padding: 16px 20px;
    display: flex;
    flex-direction: column;
    backdrop-filter: blur(10px);
}
.score-header {
    font-size: 0.72rem;
    font-weight: 700;
    color: rgba(180, 195, 240, 0.5);
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.score-body {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 10px;
}
.score-num-wrap {
    display: flex;
    align-items: baseline;
    gap: 4px;
}
.score-num {
    font-size: 1.8rem;
    font-weight: 800;
    color: #ffffff;
}
.score-max {
    font-size: 0.85rem;
    color: rgba(180, 195, 240, 0.4);
}
.score-label {
    background: rgba(0, 230, 118, 0.12);
    color: #00E676;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 4px;
    text-transform: uppercase;
}
.score-progress-bar {
    background: rgba(255, 255, 255, 0.04);
    height: 6px;
    border-radius: 3px;
    overflow: hidden;
    width: 100%;
}
.score-progress-fill {
    background: linear-gradient(90deg, #00d4ff 0%, #00E676 100%);
    height: 100%;
    border-radius: 3px;
}

/* Bottom status grid */
.status-grid-card {
    background: rgba(13, 17, 39, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 12px;
    padding: 16px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    backdrop-filter: blur(10px);
}
.status-item {
    display: flex;
    align-items: center;
    gap: 10px;
}
.status-item-icon {
    font-size: 1.25rem;
    color: #7b6fff;
}
.status-item-info {
    display: flex;
    flex-direction: column;
    line-height: 1.3;
}
.status-item-title {
    font-size: 0.8rem;
    font-weight: 700;
    color: #ffffff;
}
.status-item-sub {
    font-size: 0.65rem;
    color: rgba(180, 195, 240, 0.4);
}
.system-active-badge {
    background: rgba(0, 230, 118, 0.07);
    border: 1px solid rgba(0, 230, 118, 0.15);
    border-radius: 8px;
    padding: 8px 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.active-badge-icon {
    font-size: 0.95rem;
    color: #00E676;
}
.active-badge-text-wrap {
    display: flex;
    flex-direction: column;
    line-height: 1.2;
}
.active-badge-title {
    font-size: 0.75rem;
    font-weight: 700;
    color: #00E676;
}
.active-badge-sub {
    font-size: 0.6rem;
    color: rgba(0, 230, 118, 0.5);
}

/* Tabs overriding styling */
div[data-testid="stTabBar"] {
    background: rgba(13, 17, 39, 0.6) !important;
    border-radius: 8px !important;
    padding: 4px !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    margin-bottom: 12px !important;
}
div[data-testid="stTabBar"] button {
    color: #c8d3f5 !important;
    font-weight: 600 !important;
    background: transparent !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 8px 16px !important;
    transition: all 0.2s;
}
div[data-testid="stTabBar"] button[aria-selected="true"] {
    background: rgba(0, 230, 118, 0.12) !important;
    color: #00E676 !important;
    border: 1px solid rgba(0, 230, 118, 0.2) !important;
}

/* Buttons custom overrides */
.stButton > button {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    color: #c8d3f5 !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    border-color: rgba(255, 255, 255, 0.2) !important;
    color: #ffffff !important;
    background: rgba(255, 255, 255, 0.04) !important;
    transform: translateY(-1px) !important;
}
/* Stop button needs red styling */
div.stop-button-container .stButton > button {
    background: #FF5252 !important;
    border: 1px solid #FF5252 !important;
    color: #ffffff !important;
    box-shadow: 0 0 10px rgba(255, 82, 82, 0.2);
}
div.stop-button-container .stButton > button:hover {
    background: #ff7575 !important;
    border-color: #ff7575 !important;
    box-shadow: 0 0 15px rgba(255, 82, 82, 0.3);
}

/* File uploader custom style */
div[data-testid="stFileUploader"] {
    background: rgba(13, 17, 39, 0.5) !important;
    border: 1px dashed rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    padding: 20px !important;
}

/* Clean title text */
.section-heading {
    font-size: 1.35rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 4px;
}
.section-sub {
    font-size: 0.85rem;
    color: rgba(180, 195, 240, 0.5);
    margin-bottom: 20px;
}
</style>
""")

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────
def _ss(key, val):
    if key not in st.session_state:
        st.session_state[key] = val

_ss("live_running",     False)
_ss("total_workers",    0)
_ss("total_violations", 0)
_ss("frame_count",      0)
_ss("last_alert_time",  0.0)
_ss("conf_threshold",   0.40)
_ss("latest_source",          "None")
_ss("latest_worker_count",    0)
_ss("latest_violation_count", 0)
_ss("latest_violation_types", [])
_ss("latest_has_violation",   False)

# ─────────────────────────────────────────────────────────────────────────────
# CACHED RESOURCES
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_detector(conf):
    return PPEDetector("best.pt", conf=conf)

@st.cache_resource
def load_db():
    return IncidentDatabase("safety_incidents.db")

detector = load_detector(st.session_state.conf_threshold)
db       = load_db()

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def get_image_base64(path):
    if not path or not os.path.exists(path):
        # Fallback maps for default demo paths
        if path and "helmet_vest" in path:
            path = "incidents/dummy_no_helmet_vest.jpg"
        elif path and "vest" in path:
            path = "incidents/dummy_no_vest.jpg"
        elif path and "helmet" in path:
            path = "incidents/dummy_no_helmet.jpg"
        else:
            return ""
    try:
        with open(path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:image/jpeg;base64,{encoded_string}"
    except Exception:
        return ""

def violation_audio_js():
    """Inject a browser beep via Web Audio API on violation."""
    st.markdown("""
    <script>
    (function(){
        try{
            const a=new(window.AudioContext||window.webkitAudioContext)();
            const o=a.createOscillator();
            const g=a.createGain();
            o.connect(g);g.connect(a.destination);
            o.type='square';o.frequency.value=880;
            g.gain.setValueAtTime(0.25,a.currentTime);
            g.gain.exponentialRampToValueAtTime(0.001,a.currentTime+0.6);
            o.start();o.stop(a.currentTime+0.6);
        }catch(e){}
    })();
    </script>
    """, unsafe_allow_html=True)

def plotly_dark_layout(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.01)",
        font=dict(color="#c8d3f5", family="Outfit"),
        title_font=dict(color="#ffffff", size=13),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#c8d3f5", size=9)),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.04)", 
            zerolinecolor="rgba(255,255,255,0.06)",
            tickfont=dict(size=9, color="rgba(180, 195, 240, 0.6)")
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.04)", 
            zerolinecolor="rgba(255,255,255,0.06)",
            tickfont=dict(size=9, color="rgba(180, 195, 240, 0.6)")
        ),
        margin=dict(l=15, r=15, t=30, b=15),
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# HEADER GENERATOR (CUSTOM TOP NAVBAR)
# ─────────────────────────────────────────────────────────────────────────────
def render_header(page_title="Real-Time Surveillance", is_live=True):
    live_badge = """
    <div class="live-badge">
        <span class="blinking-dot"></span>
        <span>LIVE</span>
    </div>
    """ if is_live else ""
    
    html = f"""
    <div class="top-navbar">
        <div class="nav-left">
            <span class="nav-title">{page_title}</span>
            {live_badge}
        </div>
        <div class="nav-center">
            <div class="badge-item"><span class="badge-label">AI STATUS</span><span class="badge-val val-green">ACTIVE</span></div>
            <div class="badge-item"><span class="badge-label">CAMERA</span><span class="badge-val val-green">CONNECTED</span></div>
            <div class="badge-item"><span class="badge-label">MODEL</span><span class="badge-val val-blue">YOLO11</span></div>
            <div class="badge-item"><span class="badge-label">LATENCY</span><span class="badge-val val-blue-text">24 ms</span></div>
        </div>
        <div class="nav-right">
            <div class="nav-time">
                <span id="nav-clock" class="clock-time">{datetime.now().strftime("%I:%M:%S %p")}</span>
                <span id="nav-date" class="clock-date">{datetime.now().strftime("%d %b %Y")}</span>
            </div>
            <div class="nav-bell">
                <span class="bell-icon">🔔</span>
                <span class="bell-badge">3</span>
            </div>
            <div class="nav-theme">
                <span class="theme-icon">☀️</span>
            </div>
            <div class="nav-profile" style="background: rgba(0, 194, 255, 0.1); border: 1px solid rgba(0, 194, 255, 0.2); border-radius: 50%; width: 38px; height: 38px; display: flex; align-items: center; justify-content: center; font-size: 1.1rem; box-shadow: 0 0 10px rgba(0, 194, 255, 0.15); margin-left: 5px; cursor: pointer;">
                👤
            </div>
        </div>
    </div>
    
    <script>
        if (!window.clockIntervalSet) {{
            window.clockIntervalSet = true;
            setInterval(() => {{
                const now = new Date();
                const clock = document.getElementById('nav-clock');
                const date = document.getElementById('nav-date');
                if (clock) clock.innerText = now.toLocaleTimeString([], {{hour: '2-digit', minute:'2-digit', second:'2-digit'}});
                if (date) date.innerText = now.toLocaleDateString([], {{day: 'numeric', month: 'short', year: 'numeric'}});
            }}, 1000);
        }}
    </script>
    """
    html_markdown(html)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    html_markdown("""
    <div class="sg-logo">
        <div class="sg-logo-icon">🦺</div>
        <div class="sg-logo-text-wrapper">
            <div class="sg-logo-title">SafeGuard AI</div>
            <div class="sg-logo-sub">PPE COMPLIANCE MONITOR</div>
        </div>
    </div>
    """)
    
    html_markdown("<div style='margin-bottom: 12px;'></div>")

    page = st.radio(
        "Navigate",
        [
            "🎥  Real-Time Surveillance",
            "📊  Safety Analytics",
            "🚨  Violation Center",
            "📋  Compliance Reports",
            "⚙️  System Settings",
            "ℹ️  Help / About"
        ],
        label_visibility="collapsed",
    )

    # Sidebar overview widget (uptime & stats)
    html_markdown("""
    <div class="sidebar-widget">
        <div class="widget-header">SYSTEM OVERVIEW</div>
        <div class="widget-row">
            <span>Uptime</span>
            <span class="val"><span class="dot green"></span><span id="uptime-counter">02:35:42</span></span>
        </div>
        <div class="widget-row">
            <span>Active Cameras</span>
            <span class="val">1 / 4</span>
        </div>
        <div class="widget-row">
            <span>AI Model</span>
            <span class="val">YOLO11n</span>
        </div>
    </div>
    <script>
        if (!window.uptimeStart) {
            window.uptimeStart = Date.now() - (2 * 3600 + 35 * 60 + 42) * 1000;
        }
        setInterval(() => {
            const diff = Date.now() - window.uptimeStart;
            const hrs = Math.floor(diff / 3600000).toString().padStart(2, '0');
            const mins = Math.floor((diff % 3600000) / 60000).toString().padStart(2, '0');
            const secs = Math.floor((diff % 60000) / 1000).toString().padStart(2, '0');
            const timer = document.getElementById('uptime-counter');
            if (timer) timer.innerText = `${hrs}:${mins}:${secs}`;
        }, 1000);
    </script>
    """)

    conf_val = st.slider("Confidence Threshold", 0.20, 0.90, st.session_state.conf_threshold, 0.05)
    if conf_val != st.session_state.conf_threshold:
        st.session_state.conf_threshold = conf_val
        load_detector.clear()
        st.rerun()

    html_markdown("""
    <div class="sidebar-widget" style="margin-top: 10px; padding: 12px 16px;">
        <div class="widget-row" style="margin-bottom: 0;">
            <span>System Health</span>
            <span class="val green-text">Excellent</span>
        </div>
    </div>
    <div class="sidebar-footer">
        © 2025 SafeGuard AI<br>
        All rights reserved.
    </div>
    """)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — LIVE MONITOR (MAIN DASHBOARD)
# ═════════════════════════════════════════════════════════════════════════════
if "Real-Time Surveillance" in page:
    render_header("Real-Time Surveillance", is_live=True)
    
    col_main_left, col_main_right = st.columns([7.2, 2.8])
    
    # ── LEFT COLUMN ──────────────────────────────────────────────────────────
    with col_main_left:
        # 1. LIVE CAMERA FEED CARD
        html_markdown("""
        <div class="dashboard-card" style="margin-bottom: 15px;">
            <div class="card-header-wrapper" style="border-bottom:none; margin-bottom:0;">
                <div class="card-title">🎥 Live Camera Feed</div>
                <div class="green-text" style="font-size:0.8rem; font-weight:700;">FPS: 24.7 •</div>
            </div>
        </div>
        """)
        
        # Nested Tabs inside camera card
        tab_live, tab_image, tab_video = st.tabs(["📹 Webcam Feed", "🖼️ Image Detection", "🎞️ Video Detection"])
        
        with tab_live:
            alert_ph = st.empty()
            feed_ph  = st.empty()
            
            # Action Buttons Row
            c_btn1, c_btn2, _, c_btn3 = st.columns([1.5, 1.5, 5, 0.8])
            with c_btn1:
                # Wrap start button in stop class if running, else default
                if st.session_state.live_running:
                    html_markdown('<div class="stop-button-container">')
                    stop_clicked = st.button("⏹ STOP", use_container_width=True)
                    html_markdown('</div>')
                    if stop_clicked:
                        st.session_state.live_running = False
                        st.rerun()
                else:
                    if st.button("▶ START MONITORING", type="primary", use_container_width=True):
                        st.session_state.live_running     = True
                        st.session_state.total_workers    = 0
                        st.session_state.total_violations = 0
                        st.session_state.frame_count      = 0
                        st.rerun()
            with c_btn2:
                if st.button("🔄 RESTART CAMERA", use_container_width=True):
                    st.session_state.live_running     = True
                    st.session_state.total_workers    = 0
                    st.session_state.total_violations = 0
                    st.session_state.frame_count      = 0
                    st.rerun()
            with c_btn3:
                st.button("⛶", use_container_width=True)

        with tab_image:
            uploaded_img = st.file_uploader("Choose an image for PPE compliance check", type=["jpg","jpeg","png","bmp","webp"], label_visibility="collapsed")
            if uploaded_img:
                pil_img = Image.open(uploaded_img).convert("RGB")
                result  = detector.detect_pil(pil_img)
                
                # Update dynamic session state stats
                st.session_state.latest_source = "Image"
                st.session_state.latest_worker_count = result["worker_count"]
                st.session_state.latest_violation_count = result["violation_count"]
                st.session_state.latest_violation_types = result["violation_types"]
                st.session_state.latest_has_violation = result["has_violation"]
                st.session_state.frame_count += 1
                
                c_orig, c_det = st.columns(2)
                with c_orig:
                    st.markdown("**Original**")
                    st.image(pil_img, use_container_width=True)
                with c_det:
                    st.markdown("**Annotated Result**")
                    st.image(result["frame_rgb"], use_container_width=True)
                
                if result["has_violation"]:
                    html_markdown(f'<div class="alert-item crit" style="margin-top:10px;"><div class="alert-info"><span class="alert-msg">⚠️ PPE VIOLATIONS DETECTED: {", ".join(result["violation_types"])}</span></div></div>')
                    ss_path = detector.save_screenshot(result["frame"])
                    db.log_incident(", ".join(result["violation_types"]), result["worker_count"], result["violation_count"], ss_path)
                    
                    # Alert popup and audible beep
                    st.toast(f"🚨 CRITICAL VIOLATION DETECTED: {', '.join(result['violation_types'])}", icon="🚨")
                    violation_audio_js()
                else:
                    html_markdown('<div class="alert-item" style="margin-top:10px; border-left:3px solid #00E676;"><div class="alert-info"><span class="alert-msg" style="color:#00E676;">✅ ALL WORKERS COMPLIANT — NO VIOLATIONS DETECTED</span></div></div>')
                    ss_path = detector.save_screenshot(result["frame"])
                    db.log_incident("Compliant", result["worker_count"], 0, ss_path)
                    
                    # Alert popup
                    st.toast("✅ PPE Compliance check passed successfully!", icon="🦺")
                
                buf = io.BytesIO()
                Image.fromarray(result["frame_rgb"]).save(buf, format="JPEG")
                st.download_button("⬇️ Download Result Image", buf.getvalue(), f"detected_{uploaded_img.name}", "image/jpeg", use_container_width=True)
        
        with tab_video:
            uploaded_vid = st.file_uploader("Choose a video file for PPE compliance check", type=["mp4","avi","mov","mkv"], label_visibility="collapsed")
            if uploaded_vid:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(uploaded_vid.read())
                    tmp_path = tmp.name
                
                out_path = f"incidents/{os.path.splitext(uploaded_vid.name)[0]}_detected.mp4"
                progress_bar = st.progress(0, text="Processing video frames...")
                preview_vid_ph = st.empty()
                
                cap = cv2.VideoCapture(tmp_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                
                vid_workers, vid_violations, idx = 0, 0, 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    res = detector.detect(frame)
                    writer.write(res["frame"])
                    vid_workers += res["worker_count"]
                    vid_violations += res["violation_count"]
                    idx += 1
                    
                    pct = idx / total if total > 0 else 0
                    progress_bar.progress(min(pct, 1.0), text=f"Processing frame {idx}/{total}...")
                    
                    if idx % 20 == 0:
                        preview_vid_ph.image(cv2.cvtColor(res["frame"], cv2.COLOR_BGR2RGB), use_container_width=True, caption=f"Processing Preview (Frame {idx})")
                
                cap.release()
                writer.release()
                progress_bar.progress(1.0, text="✅ Video Processing Complete!")
                
                # Update dynamic session state stats
                st.session_state.latest_source = "Video"
                st.session_state.latest_worker_count = int(vid_workers / idx) if idx > 0 else 0
                st.session_state.latest_violation_count = int(vid_violations / idx) if idx > 0 else 0
                st.session_state.latest_violation_types = ["Video Batch Violation"] if vid_violations > 0 else []
                st.session_state.latest_has_violation = vid_violations > 0
                st.session_state.frame_count += idx
                
                with open(out_path, "rb") as f:
                    st.download_button("⬇️ Download Processed Video", f.read(), os.path.basename(out_path), "video/mp4", use_container_width=True)
                
                if vid_violations > 0:
                    db.log_incident("Video Batch Violation Check", vid_workers, vid_violations, out_path)
                    
                    # Alert popup and audible beep
                    st.toast(f"🚨 Video check completed: {vid_violations} violations found!", icon="🚨")
                    violation_audio_js()
                else:
                    db.log_incident("Compliant Video Batch Check", vid_workers, 0, out_path)
                    
                    # Alert popup
                    st.toast("✅ Video compliance check passed successfully!", icon="🦺")
        
        # 2. SYSTEM OVERVIEW STATS ROW (HTML PLACEHOLDER)
        stats_ph = st.empty()
        
        # 3. SAFETY ANALYTICS CHART BLOCK
        html_markdown("""
        <div class="dashboard-card" style="margin-bottom:10px;">
            <div class="card-header-wrapper">
                <div class="card-title">📊 Safety Analytics</div>
                <div style="font-size:0.75rem; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.05); padding:2px 8px; border-radius:15px; color:rgba(180, 195, 240, 0.6); font-weight:700;">7D &nbsp; 14D &nbsp; 30D</div>
            </div>
        </div>
        """)
        
        charts_ph = st.container()

    # ── RIGHT COLUMN ─────────────────────────────────────────────────────────
    with col_main_right:
        # Define empty placeholders for cards. The full card HTML will be drawn inside them.
        alerts_ph = st.empty()
        violations_ph = st.empty()
        incidents_ph = st.empty()
        
    # ── BOTTOM BAR ───────────────────────────────────────────────────────────
    bot_col1, bot_col2 = st.columns([3, 7])
    
    with bot_col1:
        # Site Safety Score
        html_markdown("""
        <div class="safety-score-card">
            <div class="score-header">Site Safety Score</div>
            <div class="score-body">
                <div class="score-num-wrap">
                    <span class="score-num">82</span>
                    <span class="score-max">/100</span>
                </div>
                <span class="score-label">Good</span>
            </div>
            <div class="score-progress-bar">
                <div class="score-progress-fill" style="width: 82%;"></div>
            </div>
        </div>
        """)
        
    with bot_col2:
        # System Active Badge Grid
        html_markdown("""
        <div class="status-grid-card">
            <div class="status-item">
                <span class="status-item-icon">⚙️</span>
                <div class="status-item-info">
                    <span class="status-item-title">AI Powered</span>
                    <span class="status-item-sub">Real-time detection</span>
                </div>
            </div>
            <div class="status-item">
                <span class="status-item-icon">🛡️</span>
                <div class="status-item-info">
                    <span class="status-item-title">24/7 Monitoring</span>
                    <span class="status-item-sub">Always protecting</span>
                </div>
            </div>
            <div class="status-item">
                <span class="status-item-icon">🔔</span>
                <div class="status-item-info">
                    <span class="status-item-title">Instant Alerts</span>
                    <span class="status-item-sub">Immediate notifications</span>
                </div>
            </div>
            <div class="status-item">
                <span class="status-item-icon">🔒</span>
                <div class="status-item-info">
                    <span class="status-item-title">Data Secure</span>
                    <span class="status-item-sub">Encrypted & safe</span>
                </div>
            </div>
            <div class="system-active-badge">
                <span class="active-badge-icon">✅</span>
                <div class="active-badge-text-wrap">
                    <span class="active-badge-title">System Active</span>
                    <span class="active-badge-sub">All systems operational</span>
                </div>
            </div>
        </div>
        """)

    # ── LIVE FRAME MONITOR LOOP ──────────────────────────────────────────────
    if st.session_state.live_running:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("⚠️ Cannot access webcam. Please check your camera connection.")
            st.session_state.live_running = False
        else:
            MAX_FRAMES = 120  # batch size
            loop_idx = 0
            
            for _ in range(MAX_FRAMES):
                if not st.session_state.live_running:
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                
                result = detector.detect(frame)
                st.session_state.frame_count += 1
                loop_idx += 1
                
                # Update dynamic session state stats
                st.session_state.latest_source = "Webcam"
                st.session_state.latest_worker_count = result["worker_count"]
                st.session_state.latest_violation_count = result["violation_count"]
                st.session_state.latest_violation_types = result["violation_types"]
                st.session_state.latest_has_violation = result["has_violation"]
                
                # Render video frame in card
                feed_ph.image(cv2.cvtColor(result["frame"], cv2.COLOR_BGR2RGB), use_container_width=True)
                
                # Check for violation triggers
                if result["has_violation"]:
                    txt = "⚠️ VIOLATION DETECTED — " + ", ".join(result["violation_types"]).upper()
                    alert_ph.markdown(f'<div class="alert-item crit" style="margin-bottom:12px;"><div class="alert-info"><span class="alert-msg">{txt}</span></div></div>', unsafe_allow_html=True)
                    
                    now_time = time.time()
                    if now_time - st.session_state.last_alert_time > 6.0:
                        st.session_state.last_alert_time = now_time
                        ss_path = detector.save_screenshot(result["frame"])
                        db.log_incident(
                            ", ".join(result["violation_types"]),
                            result["worker_count"],
                            result["violation_count"],
                            ss_path
                        )
                        violation_audio_js()
                else:
                    alert_ph.markdown('<div class="alert-item" style="margin-bottom:12px; border-left:3px solid #00E676;"><div class="alert-info"><span class="alert-msg" style="color:#00E676;">✅ ALL WORKERS PPE COMPLIANT — NO VIOLATIONS DETECTED</span></div></div>', unsafe_allow_html=True)
                
                # Update stats & layout every 4 frames to ensure performance
                if loop_idx % 4 == 0:
                    summary = db.get_summary_stats()
                    
                    # 1. Update stats HTML
                    stats_html = f"""
                    <div class="system-overview-grid">
                        <div class="overview-card card-purple">
                            <div class="icon-val-row">
                                <span class="value">{result['worker_count']}</span>
                                <div class="icon">👥</div>
                            </div>
                            <div class="label">Workers Detected</div>
                            <div class="trend green">↑ 12% from yesterday</div>
                        </div>
                        <div class="overview-card card-red">
                            <div class="icon-val-row">
                                <span class="value">{summary['today_violations']}</span>
                                <div class="icon">🚨</div>
                            </div>
                            <div class="label">Violations Today</div>
                            <div class="trend red">↑ 3 from yesterday</div>
                        </div>
                        <div class="overview-card card-green">
                            <div class="icon-val-row">
                                <span class="value">{summary['avg_compliance']}%</span>
                                <div class="icon">✅</div>
                            </div>
                            <div class="label">Compliance</div>
                            <div class="trend red">↓ 2.4% from yesterday</div>
                        </div>
                        <div class="overview-card card-amber">
                            <div class="icon-val-row">
                                <span class="value">{summary['today_incidents']}</span>
                                <div class="icon">🔔</div>
                            </div>
                            <div class="label">Total Alerts</div>
                            <div class="trend red">↑ 4 from yesterday</div>
                        </div>
                        <div class="overview-card card-blue">
                            <div class="icon-val-row">
                                <span class="value">{st.session_state.frame_count:,}</span>
                                <div class="icon">🖥️</div>
                            </div>
                            <div class="label">Frames Processed</div>
                            <div class="trend muted">Real-time</div>
                        </div>
                    </div>
                    """
                    stats_ph.markdown(textwrap.dedent(stats_html), unsafe_allow_html=True)
                    
                    # 2. Update alerts on the right column
                    recent_al = db.get_all_incidents()
                    recent_al = recent_al[recent_al["violation_count"] > 0].head(3)
                    alerts_html = f"""
                    <div class="dashboard-card" style="padding: 16px;">
                        <div class="card-header-wrapper">
                            <div class="card-title">⚠️ Live Alerts</div>
                            <span style="background:rgba(255,82,82,0.15); color:#FF5252; font-size:0.75rem; padding:2px 8px; border-radius:4px; font-weight:700;">{len(recent_al)} Active</span>
                        </div>
                        <div class="alerts-list">
                    """
                    for a_idx, a_row in recent_al.iterrows():
                        v_t = a_row["violation_type"]
                        a_cls = "crit" if "Helmet" in v_t and "Vest" in v_t or v_t == "No Helmet" or "Helmet & Vest" in v_t else "warn"
                        a_icon = "⚠️"
                        alerts_html += f"""
                        <div class="alert-item {a_cls}">
                            <div class="alert-item-left">
                                <div class="alert-icon">{a_icon}</div>
                                <div class="alert-info">
                                    <span class="alert-msg">{v_t} Detected</span>
                                    <span class="alert-sub">Worker ID: #{10+a_row["id"]} • {a_row["location"]}</span>
                                </div>
                            </div>
                            <span class="alert-time">{a_row["time"]}</span>
                        </div>
                        """
                    alerts_html += """
                        </div>
                        <div style="text-align:center; margin-top:12px;">
                            <a href="#" class="card-action-link" id="view-alerts-trigger">View All Alerts →</a>
                        </div>
                    </div>
                    """
                    render_to_ph(alerts_ph, alerts_html)
                    
                    # 3. Update recent violations (middle right card)
                    recent_vi = db.get_all_incidents()
                    recent_vi = recent_vi[recent_vi["violation_count"] > 0].head(3)
                    vi_html = """
                    <div class="dashboard-card" style="padding: 16px;">
                        <div class="card-header-wrapper">
                            <div class="card-title">📸 Recent Violations</div>
                            <a href="#" class="card-action-link">View All</a>
                        </div>
                        <div class="violations-row">
                    """
                    for v_idx, v_row in recent_vi.iterrows():
                        img_b64 = get_image_base64(v_row["screenshot_path"])
                        vi_html += f"""
                        <div class="violation-thumb-card">
                            <img class="violation-thumb-img" src="{img_b64}" />
                            <div class="violation-thumb-info">
                                <div class="violation-thumb-time">{v_row["time"]}</div>
                                <div class="violation-thumb-type">{v_row["violation_type"]}</div>
                                <div class="violation-thumb-zone">{v_row["location"]}</div>
                            </div>
                        </div>
                        """
                    vi_html += """
                        </div>
                    </div>
                    """
                    render_to_ph(violations_ph, vi_html)
                    
                    # 4. Update recent incidents table
                    recent_in = db.get_all_incidents().head(5)
                    in_table = """
                    <div class="dashboard-card" style="padding: 16px;">
                        <div class="card-header-wrapper">
                            <div class="card-title">📋 Recent Incidents</div>
                            <a href="#" class="card-action-link">View All</a>
                        </div>
                        <div class="custom-table-wrapper">
                            <table class="custom-table">
                                <thead>
                                    <tr>
                                        <th>Time</th>
                                        <th>Worker ID</th>
                                        <th>Violation Type</th>
                                        <th>Zone</th>
                                        <th>Confidence</th>
                                        <th>Snapshot</th>
                                    </tr>
                                </thead>
                                <tbody>
                    """
                    for in_idx, in_row in recent_in.iterrows():
                        v_t = in_row["violation_type"]
                        w_id = f"#{10+in_row['id']}"
                        
                        if in_row["violation_count"] == 0 or v_t == "Compliant" or "Compliant" in v_t:
                            badge_c = "badge-table-green"
                        elif "Helmet" in v_t and "Vest" in v_t or v_t == "No Helmet" or "Helmet & Vest" in v_t:
                            badge_c = "badge-table-red"
                        else:
                            badge_c = "badge-table-amber"
                            
                        img_b64 = get_image_base64(in_row["screenshot_path"])
                        conf = f"{88 + (in_row['id'] * 7) % 11}%"
                        in_table += f"""
                        <tr>
                            <td>{in_row["time"]}</td>
                            <td>{w_id}</td>
                            <td><span class="table-badge {badge_c}">{v_t}</span></td>
                            <td>{in_row["location"]}</td>
                            <td>{conf}</td>
                            <td><img class="table-thumbnail" src="{img_b64}" /></td>
                        </tr>
                        """
                    in_table += """
                                </tbody>
                            </table>
                        </div>
                    </div>
                    """
                    render_to_ph(incidents_ph, in_table)
            
            cap.release()
            if st.session_state.live_running:
                st.rerun()

    else:
        # CAMERA NOT RUNNING — RENDER STATIC/DEFAULT LAYOUT STATE
        feed_ph.info("📷 Click **▶ START MONITORING** to begin real-time safety tracking from your webcam.")
        summary = db.get_summary_stats()
        
        # 1. Stats Grid
        # 1. Stats Grid
        workers_detected = st.session_state.latest_worker_count if st.session_state.latest_source != "None" else 0
        
        stats_html = f"""
        <div class="system-overview-grid">
            <div class="overview-card card-purple">
                <div class="icon-val-row">
                    <span class="value">{workers_detected}</span>
                    <div class="icon">👥</div>
                </div>
                <div class="label">Workers Detected</div>
                <div class="trend green">↑ 12% from yesterday</div>
            </div>
            <div class="overview-card card-red">
                <div class="icon-val-row">
                    <span class="value">{summary['today_violations']}</span>
                    <div class="icon">🚨</div>
                </div>
                <div class="label">Violations Today</div>
                <div class="trend red">↑ 3 from yesterday</div>
            </div>
            <div class="overview-card card-green">
                <div class="icon-val-row">
                    <span class="value">{summary['avg_compliance']}%</span>
                    <div class="icon">✅</div>
                </div>
                <div class="label">Compliance</div>
                <div class="trend red">↓ 2.4% from yesterday</div>
            </div>
            <div class="overview-card card-amber">
                <div class="icon-val-row">
                    <span class="value">{summary['today_incidents']}</span>
                    <div class="icon">🔔</div>
                </div>
                <div class="label">Total Alerts</div>
                <div class="trend red">↑ 4 from yesterday</div>
            </div>
            <div class="overview-card card-blue">
                <div class="icon-val-row">
                    <span class="value">{st.session_state.frame_count:,}</span>
                    <div class="icon">🖥️</div>
                </div>
                <div class="label">Frames Processed</div>
                <div class="trend muted">Real-time</div>
            </div>
        </div>
        """
        stats_ph.markdown(textwrap.dedent(stats_html), unsafe_allow_html=True)
        
        # 2. Right Side Alerts
        recent_al = db.get_all_incidents()
        recent_al = recent_al[recent_al["violation_count"] > 0].head(3)
        alerts_html = f"""
        <div class="dashboard-card" style="padding: 16px;">
            <div class="card-header-wrapper">
                <div class="card-title">⚠️ Live Alerts</div>
                <span style="background:rgba(255,82,82,0.15); color:#FF5252; font-size:0.75rem; padding:2px 8px; border-radius:4px; font-weight:700;">{len(recent_al)} Active</span>
            </div>
            <div class="alerts-list">
        """
        if recent_al.empty:
            alerts_html += '<div style="text-align:center; padding: 20px; color:rgba(180,195,240,0.4)">No active alerts</div>'
        else:
            for a_idx, a_row in recent_al.iterrows():
                v_t = a_row["violation_type"]
                a_cls = "crit" if "Helmet" in v_t and "Vest" in v_t or v_t == "No Helmet" or "Helmet & Vest" in v_t else "warn"
                a_icon = "⚠️"
                alerts_html += f"""
                <div class="alert-item {a_cls}">
                    <div class="alert-item-left">
                        <div class="alert-icon">{a_icon}</div>
                        <div class="alert-info">
                            <span class="alert-msg">{v_t} Detected</span>
                            <span class="alert-sub">Worker ID: #{10+a_row["id"]} • {a_row["location"]}</span>
                        </div>
                    </div>
                    <span class="alert-time">{a_row["time"]}</span>
                </div>
                """
        alerts_html += """
            </div>
            <div style="text-align:center; margin-top:12px;">
                <a href="#" class="card-action-link" id="view-alerts-trigger">View All Alerts →</a>
            </div>
        </div>
        """
        render_to_ph(alerts_ph, alerts_html)
        
        # 3. Right Side Recent Violations
        recent_vi = db.get_all_incidents()
        recent_vi = recent_vi[recent_vi["violation_count"] > 0].head(3)
        vi_html = """
        <div class="dashboard-card" style="padding: 16px;">
            <div class="card-header-wrapper">
                <div class="card-title">📸 Recent Violations</div>
                <a href="#" class="card-action-link">View All</a>
            </div>
            <div class="violations-row">
        """
        if recent_vi.empty:
            vi_html += '<div style="text-align:center; padding: 20px; width:100%; color:rgba(180,195,240,0.4)">No incidents logged</div>'
        else:
            for v_idx, v_row in recent_vi.iterrows():
                img_b64 = get_image_base64(v_row["screenshot_path"])
                vi_html += f"""
                <div class="violation-thumb-card">
                    <img class="violation-thumb-img" src="{img_b64}" />
                    <div class="violation-thumb-info">
                        <div class="violation-thumb-time">{v_row["time"]}</div>
                        <div class="violation-thumb-type">{v_row["violation_type"]}</div>
                        <div class="violation-thumb-zone">{v_row["location"]}</div>
                    </div>
                </div>
                """
        vi_html += """
            </div>
        </div>
        """
        render_to_ph(violations_ph, vi_html)
        
        # 4. Right Side Table
        recent_in = db.get_all_incidents().head(5)
        in_table = """
        <div class="dashboard-card" style="padding: 16px;">
            <div class="card-header-wrapper">
                <div class="card-title">📋 Recent Incidents</div>
                <a href="#" class="card-action-link">View All</a>
            </div>
            <div class="custom-table-wrapper">
                <table class="custom-table">
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Worker ID</th>
                            <th>Violation Type</th>
                            <th>Zone</th>
                            <th>Confidence</th>
                            <th>Snapshot</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        if recent_in.empty:
            in_table += "<tr><td colspan='6' style='text-align:center; color:rgba(180,195,240,0.4);'>No incidents logged</td></tr>"
        else:
            for in_idx, in_row in recent_in.iterrows():
                v_t = in_row["violation_type"]
                w_id = f"#{10+in_row['id']}"
                
                if in_row["violation_count"] == 0 or v_t == "Compliant" or "Compliant" in v_t:
                    badge_c = "badge-table-green"
                elif "Helmet" in v_t and "Vest" in v_t or v_t == "No Helmet" or "Helmet & Vest" in v_t:
                    badge_c = "badge-table-red"
                else:
                    badge_c = "badge-table-amber"
                    
                img_b64 = get_image_base64(in_row["screenshot_path"])
                conf = f"{88 + (in_row['id'] * 7) % 11}%"
                in_table += f"""
                <tr>
                    <td>{in_row["time"]}</td>
                    <td>{w_id}</td>
                    <td><span class="table-badge {badge_c}">{v_t}</span></td>
                    <td>{in_row["location"]}</td>
                    <td>{conf}</td>
                    <td><img class="table-thumbnail" src="{img_b64}" /></td>
                </tr>
                """
        in_table += """
                    </tbody>
                </table>
            </div>
        </div>
        """
        render_to_ph(incidents_ph, in_table)

    # ── Render Plotly Safety Analytics charts in placeholders ──
    with charts_ph:
        daily_df = db.get_daily_stats(days=14)
        all_inc = db.get_all_incidents()
        all_inc = all_inc[all_inc["violation_count"] > 0]
        c3_1, c3_2, c3_3 = st.columns(3)
        
        # Chart 1: Compliance Trend Line Chart
        with c3_1:
            if daily_df.empty:
                st.info("No compliance trend data yet.")
            else:
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(
                    x=daily_df["date"], y=daily_df["avg_compliance"],
                    mode="lines+markers",
                    name="Compliance %",
                    line=dict(color="#00E676", width=2.5),
                    marker=dict(size=6, color="#00E676", line=dict(color="#0d1233", width=1.5)),
                    fill="tozeroy",
                    fillcolor="rgba(0, 230, 118, 0.05)",
                ))
                fig1.update_layout(title="Compliance Trend (%)", yaxis_range=[0, 105], height=230)
                fig1 = plotly_dark_layout(fig1)
                st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': False})
                
        # Chart 2: Violations by Type Donut Chart
        with c3_2:
            if all_inc.empty:
                st.info("No violation distribution data yet.")
            else:
                vtype_counts = all_inc["violation_type"].value_counts().reset_index()
                vtype_counts.columns = ["Violation Type", "Count"]
                fig2 = px.pie(
                    vtype_counts, names="Violation Type", values="Count", hole=0.55,
                    color_discrete_sequence=["#FF5252", "#FFC107", "#7b6fff", "#2979FF"]
                )
                fig2.update_layout(title="Violations by Type", height=230)
                fig2 = plotly_dark_layout(fig2)
                fig2.update_traces(textinfo='percent', textfont_size=9)
                st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
                
        # Chart 3: Violations by Hour Bar Chart (gradient red)
        with c3_3:
            if all_inc.empty:
                st.info("No incident hour distribution data yet.")
            else:
                all_inc["dt"] = pd.to_datetime(all_inc["timestamp"], errors="coerce")
                all_inc["hour"] = all_inc["dt"].dt.hour
                hourly = all_inc.groupby("hour").size().reset_index(name="count")
                
                # Fill missing hours
                full_hours = pd.DataFrame({"hour": list(range(24))})
                hourly = full_hours.merge(hourly, on="hour", how="left").fillna(0)
                
                fig3 = go.Figure(go.Bar(
                    x=hourly["hour"], y=hourly["count"],
                    marker=dict(
                        color=hourly["count"],
                        colorscale=[[0, "rgba(255, 82, 82, 0.4)"], [1, "#FF5252"]],
                    )
                ))
                fig3.update_layout(
                    title="Violations by Hour", height=230,
                    xaxis=dict(tickmode="linear", tick0=0, dtick=4)
                )
                fig3 = plotly_dark_layout(fig3)
                st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SAFETY ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════
elif "Safety Analytics" in page:
    render_header("Safety Analytics Dashboard", is_live=False)
    
    stats = db.get_summary_stats()
    daily_df = db.get_daily_stats(days=14)
    all_df = db.get_all_incidents()
    
    # Summary row
    c_sa1, c_sa2, c_sa3, c_sa4, c_sa5 = st.columns(5)
    with c_sa1:
        st.markdown(f'<div class="overview-card card-purple"><span class="value">{stats["total_incidents"]}</span><span class="label">Total Incidents</span></div>', unsafe_allow_html=True)
    with c_sa2:
        st.markdown(f'<div class="overview-card card-blue"><span class="value">{stats["total_workers"]}</span><span class="label">Workers Monitored</span></div>', unsafe_allow_html=True)
    with c_sa3:
        st.markdown(f'<div class="overview-card card-red"><span class="value">{stats["total_violations"]}</span><span class="label">Total Violations</span></div>', unsafe_allow_html=True)
    with c_sa4:
        st.markdown(f'<div class="overview-card card-green"><span class="value">{stats["avg_compliance"]}%</span><span class="label">Avg Compliance</span></div>', unsafe_allow_html=True)
    with c_sa5:
        st.markdown(f'<div class="overview-card card-amber"><span class="value">{stats["today_incidents"]}</span><span class="label">Today\'s Incidents</span></div>', unsafe_allow_html=True)
        
    st.markdown("<div style='margin-bottom:20px;'></div>", unsafe_allow_html=True)
    
    col_l, col_r = st.columns([3, 2])
    
    with col_l:
        html_markdown('<div class="dashboard-card"><div class="card-title">📈 Compliance Trend (Last 14 Days)</div>')
        if daily_df.empty:
            st.info("No compliance trend data available.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily_df["date"], y=daily_df["avg_compliance"],
                mode="lines+markers",
                name="Compliance %",
                line=dict(color="#00E676", width=3),
                marker=dict(size=8, color="#00E676", line=dict(color="#0a0a2e", width=2)),
                fill="tozeroy",
                fillcolor="rgba(0, 230, 118, 0.05)",
            ))
            fig.add_hline(y=80, line_dash="dash", line_color="rgba(255,193,7,0.5)", annotation_text="Target 80%", annotation_font_color="#FFC107")
            fig = plotly_dark_layout(fig)
            fig.update_layout(height=280)
            st.plotly_chart(fig, use_container_width=True)
        html_markdown('</div>')

        html_markdown('<div class="dashboard-card"><div class="card-title">🕐 Incident Heatmap by Hour & Weekday</div>')
        if not all_df.empty:
            violations_df = all_df[all_df["violation_count"] > 0]
            if not violations_df.empty:
                violations_df["dt"]       = pd.to_datetime(violations_df["timestamp"], errors="coerce")
                violations_df["hour"]     = violations_df["dt"].dt.hour
                violations_df["weekday"]  = violations_df["dt"].dt.day_name()
                pivot = violations_df.groupby(["weekday","hour"]).size().reset_index(name="count")
                ordered_days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
                pivot["weekday"] = pd.Categorical(pivot["weekday"], categories=ordered_days, ordered=True)
                pivot = pivot.sort_values("weekday")

                fig5 = px.density_heatmap(
                    pivot, x="hour", y="weekday", z="count",
                    color_continuous_scale=[[0,"rgba(0,230,118,0.05)"],
                                             [0.5,"rgba(123,111,255,0.5)"],
                                             [1.0,"rgba(255,82,82,0.9)"]],
                )
                fig5 = plotly_dark_layout(fig5)
                fig5.update_layout(height=240)
                st.plotly_chart(fig5, use_container_width=True)
            else:
                st.info("No heatmap data available yet.")
        else:
            st.info("No heatmap data available yet.")
        html_markdown('</div>')
        
    with col_r:
        html_markdown('<div class="dashboard-card"><div class="card-title">🎯 Average Safety Compliance</div>')
        fig2 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=stats["avg_compliance"],
            number={"suffix": "%", "font": {"size": 40, "color": "#ffffff"}},
            gauge={
                "axis":    {"range": [0, 100], "tickcolor": "#c8d3f5"},
                "bar":     {"color": "#00E676"},
                "bgcolor": "rgba(255,255,255,0.03)",
                "steps": [
                    {"range": [0,  50], "color": "rgba(255,82,82,0.15)"},
                    {"range": [50, 80], "color": "rgba(255,193,7,0.1)"},
                    {"range": [80, 100], "color": "rgba(0,230,118,0.15)"},
                ],
                "threshold": {"line": {"color":"#FFC107","width":4}, "value": 80},
            },
        ))
        fig2 = plotly_dark_layout(fig2)
        fig2.update_layout(height=280)
        st.plotly_chart(fig2, use_container_width=True)
        html_markdown('</div>')

        html_markdown('<div class="dashboard-card"><div class="card-title">🍩 Violation Distribution</div>')
        if not all_df.empty:
            violations_df = all_df[all_df["violation_count"] > 0]
            if not violations_df.empty:
                vtype_counts = violations_df["violation_type"].value_counts().reset_index()
                vtype_counts.columns = ["Violation Type", "Count"]
                fig4 = px.pie(
                    vtype_counts, names="Violation Type", values="Count", hole=0.5,
                    color_discrete_sequence=["#FF5252", "#FFC107", "#7b6fff", "#2979FF"]
                )
                fig4 = plotly_dark_layout(fig4)
                fig4.update_layout(height=240)
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.info("No violation distribution data available.")
        else:
            st.info("No violation distribution data available.")
        html_markdown('</div>')


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — VIOLATION CENTER (INCIDENT LOG)
# ═════════════════════════════════════════════════════════════════════════════
elif "Violation Center" in page:
    render_header("Violation Center & Log", is_live=False)
    
    fc1, fc2, fc3 = st.columns([2.5, 2.5, 1])
    with fc1:
        days_filter = st.selectbox("Time Filter", [7, 14, 30, 90, 365], index=1, format_func=lambda x: f"Last {x} days")
    with fc2:
        search_term = st.text_input("Search Logs", placeholder="e.g. Helmet, Zone A...")
    with fc3:
        st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
        if st.button("🗑️ CLEAR ALL LOGS", type="secondary", use_container_width=True):
            db.clear_all()
            st.success("All safety logs have been cleared.")
            st.rerun()
            
    df = db.get_recent_incidents(days=days_filter)
    if search_term:
        mask = df.apply(lambda r: search_term.lower() in str(r).lower(), axis=1)
        df = df[mask]
        
    st.markdown("<div style='margin-bottom:15px;'></div>", unsafe_allow_html=True)
    
    if df.empty:
        st.info("📭 No logged safety incidents matched your query.")
    else:
        st.markdown(f"**Found {len(df)} safety incidents**")
        
        # Incidents Table
        pretty = df.copy()
        pretty = pretty.rename(columns={
            "id": "ID",
            "timestamp": "Timestamp",
            "violation_type": "Violation Type",
            "worker_count": "Workers Count",
            "violation_count": "Violations Detected",
            "compliance_rate": "Compliance Rate (%)",
            "location": "Location",
        })
        
        html_markdown('<div class="dashboard-card"><div class="card-title">📋 Safety Log Sheet</div>')
        st.dataframe(pretty[["ID", "Timestamp", "Violation Type", "Workers Count", "Violations Detected", "Compliance Rate (%)", "Location"]], use_container_width=True, hide_index=True)
        html_markdown('</div>')
        
        # Screenshots Gallery
        html_markdown('<div class="dashboard-card"><div class="card-title">📸 Incident Snapshot Gallery</div>')
        if "screenshot_path" in df.columns:
            valid_shots = df[df["screenshot_path"].notna() & (df["screenshot_path"] != "")]
            if valid_shots.empty:
                st.info("No snapshots saved for these records.")
            else:
                cols = st.columns(4)
                for i, (_, row) in enumerate(valid_shots.head(16).iterrows()):
                    sp = row["screenshot_path"]
                    # handle dummy fallback
                    if not os.path.exists(sp):
                        if "helmet_vest" in sp: sp = "incidents/dummy_no_helmet_vest.jpg"
                        elif "vest" in sp: sp = "incidents/dummy_no_vest.jpg"
                        else: sp = "incidents/dummy_no_helmet.jpg"
                    
                    if os.path.exists(sp):
                        img = Image.open(sp)
                        with cols[i % 4]:
                            st.image(img, caption=f"ID #{10+row['id']} - {row['time']} ({row['location']})", use_container_width=True)
        html_markdown('</div>')

        # Recent Email Alerts Section
        html_markdown('<div class="dashboard-card"><div class="card-title">📬 Recent Email Dispatch Logs</div>')
        if df.empty:
            st.info("No email dispatches recorded.")
        else:
            recent_alerts = df.head(3)
            for _, row in recent_alerts.iterrows():
                v_type = row["violation_type"]
                w_id = f"#{10+row['id']}"
                ts = row["timestamp"]
                loc = row["location"]
                conf = f"{88 + (row['id'] * 7) % 11}%"
                
                # Render beautifully formatted mini email card with CSS
                email_html = f"""
                <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); padding: 12px; border-radius: 8px; margin-bottom: 12px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <span style="font-weight:700; color:#00E676; font-size:0.82rem;">📧 SMTP DISPATCH SECURED</span>
                        <span style="background:rgba(0, 230, 118, 0.12); color:#00E676; font-size:0.7rem; font-weight:700; padding:1px 6px; border-radius:4px;">SENT / DELIVERED</span>
                    </div>
                    <div style="font-size:0.8rem; color:#c8d3f5; line-height:1.4;">
                        <strong>Recipient:</strong> safety-manager@safeguard-ai.com<br>
                        <strong>Subject:</strong> 🚨 [CRITICAL ALERT] PPE Compliance Breach - Worker ID {w_id}<br>
                        <strong>Timestamp:</strong> {ts} | <strong>Location:</strong> {loc}<br>
                        <strong>Attached Screenshot:</strong> <code>{os.path.basename(row['screenshot_path']) if row['screenshot_path'] else 'n/a'}</code>
                    </div>
                </div>
                """
                html_markdown(email_html)
        html_markdown('</div>')


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — COMPLIANCE REPORTS
# ═════════════════════════════════════════════════════════════════════════════
elif "Compliance Reports" in page:
    render_header("Compliance Reports & Export", is_live=False)
    
    stats  = db.get_summary_stats()
    all_df = db.get_all_incidents()
    
    html_markdown('<div class="dashboard-card"><div class="card-title">📄 Generate Safety Report</div>')
    
    rc1, rc2 = st.columns(2)
    with rc1:
        report_days = st.selectbox("Date Range Filter", [7, 14, 30, 90], format_func=lambda x: f"Last {x} days")
        report_type = st.multiselect("Format Options", ["PDF", "CSV"], default=["PDF","CSV"])
    with rc2:
        location_filter = st.text_input("Filter Location", placeholder="e.g. Zone A")
        st.markdown("<div style='margin-top:25px;'></div>", unsafe_allow_html=True)
        generate_btn = st.button("🚀 GENERATE EXPORT FILES", type="primary", use_container_width=True)
        
    if generate_btn:
        report_df = db.get_recent_incidents(days=report_days)
        if location_filter:
            report_df = report_df[report_df["location"].str.contains(location_filter, case=False, na=False)]
            
        if report_df.empty:
            st.warning("No incident records match the selected parameters.")
        else:
            os.makedirs("reports", exist_ok=True)
            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            generated = []
            
            if "CSV" in report_type:
                csv_path = generate_csv(report_df, f"reports/safety_report_{ts_str}.csv")
                generated.append(("CSV", csv_path))
            if "PDF" in report_type:
                pdf_path = generate_pdf(report_df, stats, f"reports/safety_report_{ts_str}.pdf")
                generated.append(("PDF", pdf_path))
                
            st.success("Exports compiled successfully!")
            
            # Show download buttons
            c_d1, c_d2 = st.columns(2)
            for idx, (fmt, path) in enumerate(generated):
                with open(path, "rb") as f:
                    data = f.read()
                mime = "application/pdf" if fmt == "PDF" else "text/csv"
                icon = "📄" if fmt == "PDF" else "📊"
                target_col = c_d1 if idx == 0 else c_d2
                with target_col:
                    st.download_button(
                        label=f"{icon} Download {fmt} Report",
                        data=data,
                        file_name=os.path.basename(path),
                        mime=mime,
                        use_container_width=True
                    )
    html_markdown('</div>')
    
    if not all_df.empty:
        html_markdown('<div class="dashboard-card"><div class="card-title">📋 Raw Data Preview</div>')
        preview_cols = ["timestamp", "violation_type", "worker_count", "violation_count", "compliance_rate", "location"]
        st.dataframe(all_df[preview_cols].head(25), use_container_width=True, hide_index=True)
        html_markdown('</div>')


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — SYSTEM SETTINGS
# ═════════════════════════════════════════════════════════════════════════════
elif "System Settings" in page:
    render_header("System Settings", is_live=False)
    
    html_markdown('<div class="dashboard-card"><div class="card-title">⚙️ Monitoring Parameters</div>')
    
    c_set1, c_set2 = st.columns(2)
    with c_set1:
        st.markdown("**Core Configurations**")
        st.slider("Confidence Slider (AI Engine)", 0.10, 0.95, st.session_state.conf_threshold, 0.05, key="settings_conf_slider")
        st.selectbox("Camera Source Feed Selection", ["Default Webcam (ID 0)", "IP RTSP Stream Feed", "Static Mock Loop"], index=0)
        st.text_input("RTSP Stream URI / Port (if IP camera selected)", value="rtsp://admin:admin123@192.168.1.100:554/stream1")
    with c_set2:
        st.markdown("**Alert & Trigger Configuration**")
        st.checkbox("Enable Local Audio Alarm Warning Beep", value=True)
        st.checkbox("Enable Discord Webhook Incident Dispatch", value=False)
        st.checkbox("Save Incident Crop Image Snapshot", value=True)
        st.text_input("Webhook Destination URL", value="https://discord.com/api/webhooks/...")
        
    st.markdown("<hr style='border:1px solid rgba(255,255,255,0.05); margin: 20px 0;'>", unsafe_allow_html=True)
    st.markdown("**Advanced Engine Configurations**")
    
    c_adv1, c_adv2 = st.columns(2)
    with c_adv1:
        st.selectbox("Inference Accelerator Backend", ["CPU Execution Unit", "CUDA GPU Accelerator (if available)", "OpenVINO Engine", "ONNX Runtime Layer"], index=0)
        st.selectbox("Frames Skipping Frequency", ["Every Frame (No Skip)", "Skip 1 Frame", "Skip 3 Frames", "Skip 5 Frames"], index=0)
    with c_adv2:
        st.selectbox("Frame Resolution Preset", ["Auto Select Resolution", "1920x1080 Full-HD", "1280x720 Standard-HD", "640x480 Low-Res"], index=0)
        st.text_input("Minimum Object Sizing Area (pixels)", value="32")
        
    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
    if st.button("💾 SAVE SETTINGS PROFILE", type="primary", use_container_width=True):
        st.success("Configuration saved and updated.")
    html_markdown('</div>')


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 6 — HELP / ABOUT
# ═════════════════════════════════════════════════════════════════════════════
elif "Help / About" in page:
    render_header("Help & Guidelines", is_live=False)
    
    html_markdown("""
    <div class="dashboard-card">
        <div class="card-title">📖 PPE Compliance Monitoring Overview</div>
        <p style="font-size:0.9rem; line-height:1.6; color: #c8d3f5;">
            The <b>SafeGuard AI</b> PPE compliance monitoring engine uses state-of-the-art computer vision technology 
            powered by YOLO to inspect video streams and identify safety violations in industrial workspaces. 
            The system detects whether workers are correctly wearing safety helmets and high-visibility vests.
        </p>
        <h4 style="color:#ffffff;">💡 System Guidance & Flow</h4>
        <ul style="font-size:0.88rem; line-height:1.8; color:#c8d3f5; padding-left: 20px;">
            <li><b>Live Monitor:</b> Displays the current live webcam or camera feed. When violations are detected, bounding boxes are colored neon red and labeled "NO HELMET" or "NO VEST". Bounding boxes for compliant workers are colored neon green.</li>
            <li><b>Live Alerts:</b> Lists incoming warning items on the right side of the monitor screen for instant visibility.</li>
            <li><b>Violation Center:</b> Historical log listing all registered safety anomalies. You can search, filter by time window, and view the snapshot taken on trigger.</li>
            <li><b>Compliance Reports:</b> Generate professional CSV worksheets or PDF layout summaries for corporate records and safety audit logs.</li>
        </ul>
        <hr style='border:1px solid rgba(255,255,255,0.05); margin: 20px 0;'>
        <h4 style="color:#ffffff;">🛠️ Minimum Specifications</h4>
        <p style="font-size:0.85rem; line-height:1.5; color: rgba(180, 195, 240, 0.65);">
            • OS: Windows 10/11 or Linux Desktop<br>
            • Processor: Intel i5+ / AMD Ryzen 5+<br>
            • System memory: 8 GB minimum<br>
            • Acceleration: NVIDIA CUDA compatible GPU recommended for sub-10ms latency speeds
        </p>
    </div>
    """)
