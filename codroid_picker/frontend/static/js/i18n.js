(function () {
  window.t = function (key) {
    return (window.__I18N__ && window.__I18N__[key]) || key;
  };

  function translateTextNode(node) {
    const original = node.nodeValue;
    const trimmed = original.trim();
    if (!trimmed) return;
    const translated = window.t(trimmed);
    if (translated === trimmed) return;
    node.nodeValue = original.replace(trimmed, translated);
  }

  function translateAttribute(node, attribute) {
    const value = node.getAttribute(attribute);
    if (!value) return;
    const translated = window.t(value);
    if (translated !== value) node.setAttribute(attribute, translated);
  }

  window.hydrateI18n = function () {
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        const parent = node.parentElement;
        if (!parent || ["SCRIPT", "STYLE", "TEXTAREA", "PRE", "CODE"].includes(parent.tagName)) {
          return NodeFilter.FILTER_REJECT;
        }
        return NodeFilter.FILTER_ACCEPT;
      }
    });
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    nodes.forEach(translateTextNode);
    document.querySelectorAll("[placeholder], [title], [aria-label]").forEach((node) => {
      translateAttribute(node, "placeholder");
      translateAttribute(node, "title");
      translateAttribute(node, "aria-label");
    });
  };

  document.addEventListener("DOMContentLoaded", () => {
    window.hydrateI18n();
  });
})();
