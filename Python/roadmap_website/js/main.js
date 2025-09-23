// Requires marked via CDN, e.g.
// <script src="https://cdn.jsdelivr.net/npm/marked@4.3.0/marked.min.js"></script>

(() => {
  const INDEX_URL = 'data/weeks/index.yaml';
  const WEEK_BASE_URL = 'data/weeks/';
  const CONTENT_AREA = document.getElementById('content-area');
  const NAV_CONTAINER = document.getElementById('week-navigation');
  const SIDEBAR = document.getElementById('sidebar');
  const MENU_TOGGLE = document.getElementById('menu-toggle');
  const SIDEBAR_OVERLAY = document.getElementById('sidebar-overlay');
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

  const markedFn = window.marked || (typeof marked !== "undefined" ? marked : null);
  const yamlFn = window.jsyaml || (typeof jsyaml !== "undefined" ? jsyaml : null);

  if (!markedFn || !yamlFn) {
    console.error('Missing required parsing libraries. Ensure marked and js-yaml are loaded.');
    if (CONTENT_AREA) {
      CONTENT_AREA.innerHTML = '<div class="bg-red-50 border border-red-200 text-red-800 p-4 rounded">Configuration error: required parsers are missing.</div>';
    }
    return;
  }

  markedFn.setOptions({ gfm: true, breaks: true });

  const state = {
    weeks: [],
    cache: new Map(),
    sidebarLinks: new Map(),
    activeWeek: null,
  };

  const patternUrl = /https?:\/\/[^\s<]+/g;

  const openSidebar = () => {
    SIDEBAR.classList.remove('-translate-x-full');
    SIDEBAR_OVERLAY.classList.remove('hidden');
    requestAnimationFrame(() => SIDEBAR_OVERLAY.classList.remove('opacity-0'));
  };

  const closeSidebar = () => {
    SIDEBAR.classList.add('-translate-x-full');
    SIDEBAR_OVERLAY.classList.add('opacity-0');
    setTimeout(() => SIDEBAR_OVERLAY.classList.add('hidden'), 250);
  };

  const setActiveLink = (number) => {
    state.sidebarLinks.forEach((link, key) => {
      if (!link) return;
      if (Number(key) === Number(number)) {
        link.classList.add('active');
      } else {
        link.classList.remove('active');
      }
    });
  };

  const showMessage = (html) => {
    if (!CONTENT_AREA) return;
    CONTENT_AREA.innerHTML = html;
  };

  const showLoading = () => showMessage('<div class="text-center text-gray-500 py-12">Loading...</div>');
  const showError = (message) => showMessage(`<div class="bg-red-50 border border-red-200 text-red-800 p-4 rounded">${message}</div>`);

  const ensureAnchor = (value) => {
    if (!value) return '';
    return value.replace(patternUrl, (match) => `<a href="${match}" target="_blank" rel="noopener noreferrer">${match}</a>`);
  };

  const renderMarkdown = (markdown) => {
    if (!markdown) return '';
    return markedFn.parse(markdown);
  };

  const renderMarkdownInline = (markdown) => {
    if (!markdown) return '';
    if (typeof markedFn.parseInline === 'function') {
      return markedFn.parseInline(markdown);
    }
    return markedFn.parse(markdown);
  };

  const renderList = (items, { ordered = false } = {}) => {
    if (!items || !items.length) return '';
    const tag = ordered ? 'ol' : 'ul';
    const entries = items.map((item) => `<li>${renderMarkdownInline(item)}</li>`).join('');
    return `<${tag} class="list-disc list-inside space-y-2">${entries}</${tag}>`;
  };

  const renderKeyValueList = (title, content) => {
    if (!content) return '';
    return `<div class="space-y-2"><h5 class="text-sm font-semibold text-gray-600 uppercase tracking-wide">${title}</h5>${content}</div>`;
  };

  const extractSections = (markdown) => {
    if (!markdown) return { sections: {}, remaining: '' };
    const lines = markdown.split(/\r?\n/);
    const sections = {};
    const recognized = new Set(['Summary', 'Project Description', 'Code Focus', 'Math & Stats', 'Docs', 'Bibliography']);
    let current = null;
    let buffer = [];
    const other = [];

    const flush = () => {
      if (current) {
        sections[current] = buffer.join('\n').trim();
      } else if (buffer.length) {
        other.push(buffer.join('\n'));
      }
      buffer = [];
    };

    lines.forEach((line) => {
      const headingMatch = line.match(/^##\s+(.+)/);
      if (headingMatch) {
        flush();
        const heading = headingMatch[1].trim();
        current = recognized.has(heading) ? heading : null;
        if (!current) {
          other.push(`## ${heading}`);
        }
      } else {
        buffer.push(line);
      }
    });
    flush();

    const remaining = other.join('\n').trim();
    return { sections, remaining };
  };

  const renderBundles = (bundles) => {
    if (!bundles || !bundles.length) return '';
    return `<div class="flex flex-wrap gap-2 mt-4">${bundles.map((bundle) => `<span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">${bundle}</span>`).join('')}</div>`;
  };

  const renderObjectiveCard = (week, sections) => {
    const parts = [];
    if (sections['Summary']) {
      parts.push(renderKeyValueList('Summary', `<div class="space-y-2">${renderMarkdown(sections['Summary'])}</div>`));
    }
    if (week.project && week.project.objective) {
      parts.push(renderKeyValueList('Notebook Focus', renderMarkdownInline(week.project.objective)));
    }
    if (week.project && week.project.metrics && week.project.metrics.length) {
      parts.push(renderKeyValueList('Success Criteria', renderList(week.project.metrics)));
    }
    if (!parts.length) return '';
    return `<section class="content-card bg-white p-5 lg:col-span-2"><h4>Objective</h4><div class="mt-4 space-y-5">${parts.join('<div class="h-4"></div>')}</div></section>`;
  };

  const renderDescriptionCard = (week, sections) => {
    const parts = [];
    if (week.project && week.project.dataset) {
      parts.push(renderKeyValueList('Dataset', renderMarkdownInline(week.project.dataset)));
    }
    if (week.project && week.project.dataset_links && week.project.dataset_links.length) {
      const links = week.project.dataset_links.map((link) => renderMarkdownInline(link)).join('<br />');
      parts.push(renderKeyValueList('Dataset Links', `<div class="space-y-1">${links}</div>`));
    }
    const projectDescription = sections['Project Description'] || (week.project && week.project.description);
    if (projectDescription) {
      parts.push(renderKeyValueList('Project Description', renderMarkdown(projectDescription)));
    }
    if (week.project && week.project.nuances && week.project.nuances.length) {
      parts.push(renderKeyValueList('Nuances', renderList(week.project.nuances)));
    }
    if (week.code_focus && week.code_focus.length) {
      parts.push(renderKeyValueList('Code Focus', renderList(week.code_focus)));
    }
    if (week.math_stats && week.math_stats.length) {
      parts.push(renderKeyValueList('Math & Stats', renderList(week.math_stats)));
    }
    if (!parts.length) return '';
    return `<section class="content-card bg-white p-5 lg:col-span-2"><h4>Description</h4><div class="mt-4 space-y-5">${parts.join('<div class="h-4"></div>')}</div></section>`;
  };

  const renderResourcesCard = (week) => {
    const parts = [];
    if (week.docs && week.docs.length) {
      parts.push(renderKeyValueList('Documentation', renderList(week.docs)));
    }
    if (week.bibliography && week.bibliography.length) {
      parts.push(renderKeyValueList('Bibliography', renderList(week.bibliography)));
    }
    if (!parts.length) return '';
    return `<section class="content-card bg-white p-5 lg:col-span-2"><h4>Resources</h4><div class="mt-4 space-y-5">${parts.join('<div class="h-4"></div>')}</div></section>`;
  };

  const renderWeekView = (frontmatter, sections, remainingHtml) => {
    const headerParts = [];
    headerParts.push(`<h1 class=\"text-3xl font-bold text-gray-900\">Week ${frontmatter.number}: ${frontmatter.title}</h1>`);
    if (frontmatter.phase) {
      headerParts.push(`<p class=\"text-sm font-medium text-blue-600 uppercase tracking-wide\">${frontmatter.phase}</p>`);
    }
    const bundlesHtml = renderBundles(frontmatter.bundles);
    if (bundlesHtml) headerParts.push(bundlesHtml);
    const headerHtml = headerParts.join('');

    const objectiveCard = renderObjectiveCard(frontmatter, sections);
    const descriptionCard = renderDescriptionCard(frontmatter, sections);
    const resourcesCard = renderResourcesCard(frontmatter);
    const cardsHtml = [objectiveCard, descriptionCard, resourcesCard].filter(Boolean).join('');

    const bodyHtml = remainingHtml ? `<div class=\"prose prose-slate max-w-none\">${remainingHtml}</div>` : '';

    const wrapper = document.createElement('article');
    wrapper.className = 'space-y-8';
    wrapper.innerHTML = `
      <header class=\"space-y-3\">
        ${headerHtml}
      </header>
      <div class=\"grid grid-cols-1 gap-6\">
        ${cardsHtml}
      </div>
      ${bodyHtml}
    `;
    showMessage(wrapper.outerHTML);
  };

  const getCandidateFilenames = (number) => {
    const n = Number(number);
    const padded = `week${String(n).padStart(2, '0')}.md`;
    const plain = `week${n}.md`;
    return padded === plain ? [padded] : [padded, plain];
  };

  const fetchWeekMarkdown = async (number) => {
    const candidates = getCandidateFilenames(number);
    let lastError;
    for (const filename of candidates) {
      const url = `${WEEK_BASE_URL}${filename}`;
      try {
        const response = await fetch(url, { cache: 'no-cache' });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const text = await response.text();
        return { text, filename };
      } catch (error) {
        lastError = error;
      }
    }
    throw lastError || new Error('Week file not found');
  };

  const parseWeekMarkdown = (markdown) => {
    if (!markdown) {
      return { frontmatter: {}, sections: {}, remainingHtml: '' };
    }
    let frontmatter = {};
    let body = markdown;
    const fmMatch = markdown.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?/);
    if (fmMatch) {
      try {
        frontmatter = yamlFn.load(fmMatch[1]) || {};
      } catch (error) {
        console.warn('Failed to parse frontmatter', error);
        frontmatter = {};
      }
      body = markdown.slice(fmMatch[0].length);
    }

    const { sections, remaining } = extractSections(body);
    const remainingHtml = remaining ? renderMarkdown(remaining) : '';
    return { frontmatter, sections, remainingHtml };
  };

  const loadWeek = async (number, { pushState = true } = {}) => {
    if (!CONTENT_AREA) return;
    showLoading();
    try {
      let parsed = state.cache.get(Number(number));
      if (!parsed) {
        const { text } = await fetchWeekMarkdown(number);
        parsed = parseWeekMarkdown(text);
        parsed.frontmatter.number = parsed.frontmatter.number || Number(number);
        parsed.frontmatter.title = parsed.frontmatter.title || `Week ${number}`;
        parsed.frontmatter.phase = parsed.frontmatter.phase || '';
        parsed.frontmatter.bundles = parsed.frontmatter.bundles || [];
        parsed.frontmatter.project = parsed.frontmatter.project || {};
        parsed.frontmatter.project.metrics = parsed.frontmatter.project.metrics || [];
        parsed.frontmatter.project.nuances = parsed.frontmatter.project.nuances || [];
        parsed.frontmatter.project.dataset_links = parsed.frontmatter.project.dataset_links || [];
        parsed.frontmatter.code_focus = parsed.frontmatter.code_focus || [];
        parsed.frontmatter.math_stats = parsed.frontmatter.math_stats || [];
        parsed.frontmatter.docs = parsed.frontmatter.docs || [];
        parsed.frontmatter.bibliography = parsed.frontmatter.bibliography || [];
        state.cache.set(Number(number), parsed);
      }
      state.activeWeek = Number(number);
      setActiveLink(number);
      renderWeekView(parsed.frontmatter, parsed.sections || {}, parsed.remainingHtml || '');
      if (pushState) {
        window.location.hash = `week-${number}`;
      }
    } catch (error) {
      console.error('Failed to load week', error);
      showError(`Failed to load week ${number}. ${error.message}`);
    }
  };

  const populateSidebar = (weeks) => {
    if (!NAV_CONTAINER) return;
    NAV_CONTAINER.innerHTML = '';
    state.sidebarLinks.clear();
    weeks.forEach((week) => {
      const link = document.createElement('a');
      link.href = `#week-${week.number}`;
      link.dataset.week = week.number;
      link.className = 'sidebar-link block px-3 py-2 rounded hover:bg-blue-600/80';
      link.textContent = `Week ${week.number}: ${week.title}`;
      link.addEventListener('click', (event) => {
        event.preventDefault();
        const targetWeek = Number(link.dataset.week);
        loadWeek(targetWeek);
        if (window.innerWidth < 768) closeSidebar();
      });
      NAV_CONTAINER.appendChild(link);
      state.sidebarLinks.set(Number(week.number), link);
    });
  };

  const resolveInitialWeek = (weeks) => {
    if (!weeks.length) return null;
    const hash = window.location.hash;
    if (hash && hash.startsWith('#week-')) {
      const num = parseInt(hash.replace('#week-', ''), 10);
      if (Number.isFinite(num) && weeks.some((wk) => Number(wk.number) === num)) {
        return num;
      }
    }
    return Number(weeks[0].number);
  };

  const fetchIndex = async () => {
    try {
      const response = await fetch(INDEX_URL, { cache: 'no-cache' });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const text = await response.text();
      const parsed = yamlFn.load(text);
      const weeks = (parsed && parsed.weeks) || [];
      state.weeks = weeks.sort((a, b) => Number(a.number) - Number(b.number));
      populateSidebar(state.weeks);
      const initialWeek = resolveInitialWeek(state.weeks);
      if (initialWeek != null) {
        loadWeek(initialWeek, { pushState: false });
      } else {
        showError('No weeks available.');
      }
    } catch (error) {
      console.error('Failed to load index', error);
      showError(`Failed to load roadmap index. ${error.message}`);
    }
  };

  if (MENU_TOGGLE) {
    MENU_TOGGLE.addEventListener('click', (event) => {
      event.stopPropagation();
      if (SIDEBAR.classList.contains('-translate-x-full')) {
        openSidebar();
      } else {
        closeSidebar();
      }
    });
  }

  if (SIDEBAR_OVERLAY) {
    SIDEBAR_OVERLAY.addEventListener('click', closeSidebar);
  }

  window.addEventListener('hashchange', () => {
    const hash = window.location.hash;
    if (hash && hash.startsWith('#week-')) {
      const num = parseInt(hash.replace('#week-', ''), 10);
      if (Number.isFinite(num) && num !== state.activeWeek) {
        loadWeek(num, { pushState: false });
      }
    }
  });

  fetchIndex();
})();


