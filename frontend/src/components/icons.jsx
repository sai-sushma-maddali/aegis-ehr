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

export const LockIcon = (props) => (
  <svg {...base} {...props}>
    <rect x="4" y="10" width="16" height="10" rx="2" />
    <path d="M8 10V7a4 4 0 0 1 8 0v3" />
  </svg>
);

export const CheckCircleIcon = (props) => (
  <svg {...base} {...props}>
    <circle cx="12" cy="12" r="9" />
    <path d="m8.5 12.2 2.4 2.4 4.6-4.9" />
  </svg>
);

export const AlertTriangleIcon = (props) => (
  <svg {...base} {...props}>
    <path d="M10.6 3.5 2.9 17a1.8 1.8 0 0 0 1.6 2.7h15a1.8 1.8 0 0 0 1.6-2.7L13.4 3.5a1.8 1.8 0 0 0-2.8 0Z" />
    <path d="M12 9v4" />
    <path d="M12 16.5h.01" />
  </svg>
);

export const XCircleIcon = (props) => (
  <svg {...base} {...props}>
    <circle cx="12" cy="12" r="9" />
    <path d="m9.5 9.5 5 5" />
    <path d="m14.5 9.5-5 5" />
  </svg>
);

export const HelpCircleIcon = (props) => (
  <svg {...base} {...props}>
    <circle cx="12" cy="12" r="9" />
    <path d="M9.5 9.2a2.5 2.5 0 0 1 4.8 1c0 1.6-2.3 1.8-2.3 3.3" />
    <path d="M12 16.8h.01" />
  </svg>
);

export const ChevronRightIcon = (props) => (
  <svg {...base} {...props}>
    <path d="m9 6 6 6-6 6" />
  </svg>
);

export const ArrowRightIcon = (props) => (
  <svg {...base} {...props}>
    <path d="M4 12h16" />
    <path d="m13 5 7 7-7 7" />
  </svg>
);

export const GaugeIcon = (props) => (
  <svg {...base} {...props}>
    <circle cx="12" cy="13" r="8" />
    <path d="M12 13 15.5 9" />
    <path d="M9 4.5h6" />
  </svg>
);

export const MessageIcon = (props) => (
  <svg {...base} {...props}>
    <path d="M21 12a8 8 0 1 1-3.4-6.6" />
    <path d="M21 3v6h-6" />
  </svg>
);

export const RadarIcon = (props) => (
  <svg {...base} {...props}>
    <circle cx="12" cy="12" r="9" />
    <circle cx="12" cy="12" r="5" />
    <circle cx="12" cy="12" r="1" />
  </svg>
);

export const ActivityIcon = (props) => (
  <svg {...base} {...props}>
    <path d="M3 12h4l2 7 4-14 2 7h6" />
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

export const SettingsIcon = (props) => (
  <svg {...base} {...props}>
    <circle cx="12" cy="12" r="3" />
    <path d="M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.9-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1-1.6 1.7 1.7 0 0 0-1.9.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.9 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.6-1 1.7 1.7 0 0 0-.3-1.9l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.9.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.9-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.9V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1Z" />
  </svg>
);

export const DatabaseIcon = (props) => (
  <svg {...base} {...props}>
    <ellipse cx="12" cy="5" rx="8" ry="3" />
    <path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5" />
    <path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3" />
  </svg>
);

export const CpuIcon = (props) => (
  <svg {...base} {...props}>
    <rect x="7" y="7" width="10" height="10" rx="1" />
    <rect x="3" y="10" width="2" height="4" />
    <rect x="19" y="10" width="2" height="4" />
    <rect x="10" y="3" width="4" height="2" />
    <rect x="10" y="19" width="4" height="2" />
  </svg>
);

export const SparkleIcon = (props) => (
  <svg {...base} {...props}>
    <path d="M12 3v4" />
    <path d="M12 17v4" />
    <path d="M3 12h4" />
    <path d="M17 12h4" />
    <path d="m6 6 2.5 2.5" />
    <path d="m15.5 15.5 2.5 2.5" />
    <path d="m18 6-2.5 2.5" />
    <path d="m8.5 15.5-2.5 2.5" />
  </svg>
);
