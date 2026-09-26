// Minimal inline SVG icon set — kept dependency-free on purpose so the
// UI has no extra package to break; every icon here is a small stroke
// glyph sized to inherit `currentColor`.

const base = {
  width: 16,
  height: 16,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 2,
  strokeLinecap: "round",
  strokeLinejoin: "round",
};

export const ShieldIcon = (props) => (
  <svg {...base} {...props}>
    <path d="M12 2 4 5v6c0 5 3.4 8.7 8 11 4.6-2.3 8-6 8-11V5l-8-3Z" />
  </svg>
);

export const AlertTriangleIcon = (props) => (
  <svg {...base} {...props}>
    <path d="M10.6 3.5 2.9 17a1.8 1.8 0 0 0 1.6 2.7h15a1.8 1.8 0 0 0 1.6-2.7L13.4 3.5a1.8 1.8 0 0 0-2.8 0Z" />
    <path d="M12 9v4" />
    <path d="M12 16.5h.01" />
  </svg>
);

export const GaugeIcon = (props) => (
  <svg {...base} {...props}>
    <rect x="3" y="3" width="8" height="8" rx="1.5" />
    <rect x="13" y="3" width="8" height="5" rx="1.5" />
    <rect x="13" y="10" width="8" height="11" rx="1.5" />
    <rect x="3" y="13" width="8" height="8" rx="1.5" />
  </svg>
);

export const MessageIcon = (props) => (
  <svg {...base} {...props}>
    <path d="M21 14a3 3 0 0 1-3 3H8l-5 4V6a3 3 0 0 1 3-3h12a3 3 0 0 1 3 3z" />
    <path d="M8 9h8" />
    <path d="M8 13h5" />
  </svg>
);

export const RadarIcon = (props) => (
  <svg {...base} {...props}>
    <circle cx="12" cy="12" r="9" />
    <circle cx="12" cy="12" r="5" />
    <circle cx="12" cy="12" r="1" />
  </svg>
);

export const FileTextIcon = (props) => (
  <svg {...base} {...props}>
    <path d="M6 3h9l5 5v13H6z" />
    <path d="M14 3v5h5" />
    <path d="M9 13h6" />
    <path d="M9 17h6" />
  </svg>
);

export const PaperclipIcon = (props) => (
  <svg {...base} {...props}>
    <path d="m21.4 11.6-8.5 8.5a5 5 0 0 1-7.1-7.1l8.5-8.5a3.2 3.2 0 0 1 4.5 4.5l-8.5 8.5a1.4 1.4 0 0 1-2-2l7.8-7.8" />
  </svg>
);
