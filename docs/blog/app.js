// 阅读进度条
(function () {
  var bar = document.querySelector('.progress-bar');
  function update() {
    if (!bar) return;
    var h = document.documentElement;
    var scrolled = h.scrollTop || document.body.scrollTop;
    var height = h.scrollHeight - h.clientHeight;
    bar.style.width = (height > 0 ? (scrolled / height) * 100 : 0) + '%';
  }
  window.addEventListener('scroll', update, { passive: true });
  window.addEventListener('resize', update);
  update();
})();

// 移动端侧边栏开关
(function () {
  var btn = document.querySelector('.menu-toggle');
  var side = document.querySelector('.sidebar');
  if (!btn || !side) return;
  btn.addEventListener('click', function () {
    side.classList.toggle('open');
  });
  // 点击链接后自动收起
  side.addEventListener('click', function (e) {
    if (e.target.tagName === 'A') side.classList.remove('open');
  });
})();
