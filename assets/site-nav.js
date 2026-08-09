/* Vector Gridiron — Shared top navigation — hoops-level parity, gridiron blue theme */
(function (global) {
  'use strict';
  var LINKS = [
    { href: '/play', label: 'Play', title: 'Guess The NFL Player — fantasy style' },
    { href: '/', label: 'Cockpit', title: 'Fantasy cockpit' },
    { href: '/model', label: 'Lab', title: 'MTNN Training Cockpit + 10 Towers + Fusion' },
    { href: '/trends', label: 'Trends', title: 'Fantasy meta drift — position value shifts' },
    { href: '/players', label: 'Players', title: 'NFL player directory — skill grades + projections' },
    { href: '/methods', label: 'Methods', title: 'nflverse sources + normalization + MTNN eval' }
  ];
  function mount() {
    var nav = document.querySelector('.site-nav');
    if (!nav) return;
    var active = nav.getAttribute('data-active') || '';
    var linksHtml = LINKS.map(function (l) {
      var isActive = active === l.href ||
        (active === '/' && l.href === '/') ||
        (active === '/players' && l.href === '/players') ||
        (active === '/trends' && l.href === '/trends') ||
        (active === '/model' && l.href === '/model') ||
        (active === '/methods' && l.href === '/methods') ||
        (active === '/play' && l.href === '/play');
      return '<a class="site-nav__link' + (isActive ? ' is-active' : '') + '"' +
        ' href="' + l.href + '"' +
        (l.title ? ' title="' + l.title + '"' : '') +
        (isActive ? ' aria-current="page"' : '') +
        '>' + l.label + '</a>';
    }).join('');
    nav.innerHTML =
      '<a class="site-nav__brand" href="/">VECTOR<span class="site-nav__accent">GRIDIRON</span></a>' +
      '<div class="site-nav__links">' + linksHtml + '</div>';
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', mount);
  } else {
    mount();
  }
  global.VGSiteNav = { mount: mount, links: LINKS };
  global.VHSiteNav = global.VGSiteNav; // compat alias for shared-map etc
})(window);
