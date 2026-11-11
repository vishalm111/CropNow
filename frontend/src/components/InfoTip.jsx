import React, { useEffect, useId, useRef, useState } from 'react';

export default function InfoTip({ label, text }) {
  const [open, setOpen] = useState(false);
  const tipId = useId();
  const wrapRef = useRef(null);

  useEffect(() => {
    if (!open) return;

    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        setOpen(false);
      }
    };

    const handlePointerDown = (e) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target)) {
        setOpen(false);
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    document.addEventListener('pointerdown', handlePointerDown);
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.removeEventListener('pointerdown', handlePointerDown);
    };
  }, [open]);

  return (
    <span
      ref={wrapRef}
      className="info-tip-wrap"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onBlur={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget)) {
          setOpen(false);
        }
      }}
    >
      <button
        type="button"
        className="info-tip-btn"
        aria-label={`Explain ${label}`}
        aria-expanded={open}
        aria-controls={tipId}
        onFocus={() => setOpen(true)}
        onClick={() => setOpen((prev) => !prev)}
      >
        ?
      </button>
      {open && (
        <span id={tipId} role="tooltip" className="info-tip-bubble">
          {text}
        </span>
      )}
    </span>
  );
}
