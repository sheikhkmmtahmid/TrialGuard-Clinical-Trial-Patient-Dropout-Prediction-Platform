/**
 * TrialGuard — dashboard.js
 * Navigation, messages, IntersectionObserver fade-ins, mobile menu.
 */

(function () {
  'use strict';

  // ── Navbar scroll shadow ─────────────────────────────────────────
  const navbar = document.getElementById('navbar');
  if (navbar) {
    window.addEventListener('scroll', () => {
      if (window.scrollY > 10) {
        navbar.style.boxShadow = '0 4px 24px rgba(0,0,0,0.5)';
      } else {
        navbar.style.boxShadow = '';
      }
    }, { passive: true });
  }

  // ── Mobile hamburger ─────────────────────────────────────────────
  const hamburger = document.getElementById('navHamburger');
  const navLinks  = document.getElementById('navLinks');
  if (hamburger && navLinks) {
    hamburger.addEventListener('click', () => {
      const isOpen = navLinks.classList.toggle('open');
      hamburger.setAttribute('aria-expanded', isOpen.toString());
    });
    // Close on outside click
    document.addEventListener('click', (e) => {
      if (!navbar.contains(e.target)) {
        navLinks.classList.remove('open');
        hamburger.setAttribute('aria-expanded', 'false');
      }
    });
  }

  // ── Auto-dismiss messages ─────────────────────────────────────────
  document.querySelectorAll('.message').forEach(msg => {
    const closeBtn = msg.querySelector('.message-close');
    if (closeBtn) {
      closeBtn.addEventListener('click', () => msg.remove());
    }
    setTimeout(() => {
      msg.style.transition = 'opacity 0.4s ease, transform 0.4s ease';
      msg.style.opacity = '0';
      msg.style.transform = 'translateX(20px)';
      setTimeout(() => msg.remove(), 400);
    }, 5000);
  });

  // ── IntersectionObserver — fade-in-up sections ───────────────────
  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
          io.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.08, rootMargin: '0px 0px -40px 0px' }
  );

  document.querySelectorAll('.observe-fade').forEach(el => io.observe(el));

  // Apply observe-fade to standard animated sections (non-hero)
  document.querySelectorAll(
    '.features-grid .feature-card, .step, .tier-card, .kpi-card'
  ).forEach(el => {
    if (!el.classList.contains('fade-in') && !el.classList.contains('fade-in-up')) {
      el.classList.add('observe-fade');
      io.observe(el);
    }
  });

  // ── Risk badge pulse for Critical tier (reinforce animation) ─────
  document.querySelectorAll('.risk-critical').forEach(el => {
    el.setAttribute('title', 'CRITICAL RISK — Immediate intervention required');
  });

  // ── Highlight active nav link by current path ────────────────────
  const path = window.location.pathname;
  document.querySelectorAll('.nav-link').forEach(link => {
    const href = link.getAttribute('href');
    if (href && path.startsWith(href) && href !== '/') {
      link.classList.add('active');
    }
    if (href === '/' && path === '/') {
      link.classList.add('active');
    }
  });

  // ── Smooth scroll for anchor links ───────────────────────────────
  document.querySelectorAll('a[href^="#"]').forEach(a => {
    a.addEventListener('click', e => {
      const target = document.querySelector(a.getAttribute('href'));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });

  // ── Confirm destructive buttons ───────────────────────────────────
  document.querySelectorAll('[data-confirm]').forEach(btn => {
    btn.addEventListener('click', e => {
      if (!window.confirm(btn.dataset.confirm)) {
        e.preventDefault();
      }
    });
  });

  // ── Format probability percentages in table cells ────────────────
  document.querySelectorAll('.prob-bar').forEach(bar => {
    const pct = parseFloat(bar.style.width) || 0;
    // Colour by tier
    if (pct > 75) {
      bar.style.background = '#740001';
    } else if (pct > 55) {
      bar.style.background = '#FF5722';
    } else if (pct > 30) {
      bar.style.background = '#FFC107';
    } else {
      bar.style.background = '#4CAF50';
    }
  });

})();
