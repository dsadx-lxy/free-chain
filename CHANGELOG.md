# 更新日志

这个文件记录每一次改动。条目按 **Keep a Changelog** 的分节习惯写
（Added / Changed / Fixed），**新的写在最上面**；同一天的多批改动各占一节，
按时间倒序排列。日期是改动当天。

---

## [2026-09-22] 密钥不再随仓库和 zip 一起发出去

### Security

- **`ai_config.json` 一直被 git 跟踪**（提交 `d476125`），里面是**真实的
  DeepSeek key**。也就是说 clone 或下载这个仓库的人，拿到的不只是一份软件，
  还有一把能直接打 `api.deepseek.com` 的钥匙 —— 不需要你的机器、不需要你的
  服务在跑，就能花这份余额。`dist/` 里两个手工打的 zip（1.1.0 / 1.1.1）
  **也各带了一份**，所以不走 git、直接发 zip 一样漏。
- **止损靠吊销，不靠清理。** 历史里的那份 key 要重写 history 才抹得掉，
  但 `d476125` 早已推到远端 —— 真正有效的动作是**去 DeepSeek 后台把该 key
  吊销并重新签发**。清理历史只是打扫，不是止损。

### Changed

- `.gitignore` 从一行 `dist/` 扩成四组，并注明理由：`ai_config.json`（密钥）、
  `__pycache__/` 与 `*.pyc`（7 个字节码缓存同样曾被跟踪，约 176KB 垃圾）、
  `dist/`（打包产物）、`~$*`（Office 锁文件）。
- `git rm --cached ai_config.json` 与 `git rm -r --cached __pycache__`：
  只从索引里摘掉，**本地文件一个没动** —— 配置还在，程序照常跑。
  （对已被跟踪的文件，光写 `.gitignore` 是不起作用的，必须配这一步。）

### Added

- **`打包.py`**。手工打包没有清单、没有检查，漏一个文件不会有任何提示 ——
  而这里漏掉的恰好是最不该漏的那一个。脚本把这一步固定成三件事：
  1. 一份写在明处的排除清单（密钥 / 字节码 / 锁文件 / dist 自己）；
  2. **打完把 zip 重新打开逐条查一遍** —— 不只是「按清单排除了」，
     而是真的回去看产物里有什么；还会拿本机 key 的原文去扫包里**每一个
     文本文件**，确认它没被抄进 README、测试或别处。查出问题就**删掉 zip
     并以非零码退出**，不给「先发了再说」留缝；
  3. 打印最终清单与体积。
- 打包默认**不**放汇报 pptx（5MB，占原来下载量的 68%，与运行无关），
  要放加 `--with-slides`。版本号默认用当天日期、同名 zip 默认拒绝覆盖 ——
  代码里没有 `__version__`，历史上版本只体现在 zip 文件名上，
  猜版本正是「两个不同内容的 1.1.1.zip」的来源。

### Notes

- 用 `打包.py` 的复检对着**旧的** `dist/1.1.1.zip` 跑：报出 **11 个问题**，
  其中两条互相独立 —— `ai_config.json：密钥文件不该进包`（按文件名查）
  和 `ai_config.json：里面出现了本机密钥原文`（按内容扫）。
  就算把密钥文件改个名、或把 key 抄进别处，第二条也抓得到。
  新打的 `dist/2026-09-22.zip`（32 个文件 / 2.1 MB）复检干净。
- `server.py` 只绑 `127.0.0.1`（`app.run(host="127.0.0.1", …)`），
  所以这份 key 不会因为别人跑起服务而被反向利用 —— 它是被直接拿去打网关的。

---

## [2026-09-22] 分区导航挪到左侧 + 版式重排

### Changed

- **四个分区标签从 App Bar 里挪到左侧，排成一条竖列**（宽屏 `≥900px`）。
  点击仍然是**平滑滚动跳到那一节** —— 也就是选的那条「仍然滚动跳转」：
  页面还是一页四个 `<section>`，没有改成真的翻页，也没有上 hash 路由。
  scrollspy（滚到哪节高亮哪个、滚到底钉住最后一个）的逻辑一个字没改，
  改的只是标签往哪个方向排、排在哪儿。
  `index.html` 里**类名从 `section-tabs` 换成 `side-nav`，id 保留 `section-tabs`**
  —— `app.js` 是按 id 取的，改类名不会碰到 JS。
- **同一份标记两种形态**，靠 `@media (min-width: 900px)` 翻：
  - **≥900px**：`.viz-root` 变成 `grid-template-columns: 164px minmax(0,1fr)`，
    rail 落第一列、正文落第二列；rail 是 `position: sticky; top: 0; align-self: start`，
    所以它一路跟着滚但不把第一列撑成全高。第一列右边那条贯穿全页的竖线
    画在 **`.main-col` 的 `border-left`** 上，不是 rail 的边 —— rail 是 sticky、
    只有内容那么高，它自己的右边框只画得到那一截。
  - **<900px**：完全退回原来的顶部横条（`sticky top:0`、横向滚动、负外边距撑满）。
- **`--nav-h` 让一条规则同时服务两种形态**：JS 量出 rail 在不在侧边
  （`flexDirection === 'column'`，比比坐标稳 —— sticky 下 `getBoundingClientRect()`
  会随滚动变，flex 方向不会），rail 形态记 `0px`、横条形态记条高；
  `.app-bar { top: var(--nav-h) + var(--bar-h) }` 和
  `.page-section { scroll-margin-top: … }` 都由这两个变量拼出来，不用写两套。
  顺带补了 `ResizeObserver`（盯 bar 和 nav 本身）和 `load` 监听：抽屉开合、
  标题换行、**面板从隐藏变可见**都会改栏高，而这些一个 `resize` 事件都不发。
- **`max-width` 1440 → 1600**。rail 吃掉的是一条新「装订边」：
  164（栏）+ 1（分界线）+ 20（正文缩进）+ 20（viz 右内边距）= 205，
  原来只有 20。`1600 − 205 = 1395 ≈` 原来的 1400 —— 两张并排的图
  不会因为加了一条导航而变窄。
- **版式重排**：
  - 分区间距 20 → **48px**（左边那列把四个分区当成四个「页面」在列，间距要对得起这个语义）；
  - `.section-head` 下边距 14 → 18px，`.card-head` 10 → 12px，`.card` 底 14 → 16px；
  - `.hero` 在 ≥900px 改成两栏 grid：大数字占左列，**推导式挪到右下角、
    贴着大数字的底基线**（原来三行小字堆在左边，右边一千三百多像素几乎全空）；
  - `.tiles` 的 `minmax(178px)` → **`minmax(160px)`**；
  - `:last-child` 的收尾规则从 `.card:last-child` 放宽到 `.page-section > :last-child`
    （原来 `.tiles` 漏在外面，带着 4px 残余）。
- **侧栏当前项不再整块反白**。横条上那种小药丸搬到竖排里就是一块 140×40 的墨砖，
  压在页角上又重又空。改成 M3 导航栏的三重信号：
  **浅底 `--md-surface-container-highest` + 左侧 3px `--md-primary` 指示条 + 字重 600**，
  未选中项是 `--md-on-surface-variant` 的 500。hover 只给 `:not(.on)` 的项，
  免得鼠标移上去和「当前项」分不清。横条形态（<900px）的实心药丸**没动** ——
  小尺寸上它是对的。

### Fixed

- **侧栏吸顶时被围成一张浮在页角的卡片**。`.side-nav.stuck` 原来在所有宽度
  都吃 `--md-elev-1`，而它带 1px spread，于是 rail（左边贴页缘、右边挨竖线）
  被压出一圈完整轮廓。投影关进 `@media (max-width: 899px)`，宽屏 `box-shadow: none` ——
  侧栏是页面结构的一部分，不是浮层。窄屏那条横吸顶条仍然要投影（它下面没有竖线可分隔）。
- **rail 引入后 1440px 下统计量卡片掉成 6+1**。正文列从 1400 被挤到 1220，
  `minmax(178px)` 排不下 7 个。降到 160 后 1220 正好放 7 项一行；
  1280 视口下（内容 1060）仍是 6+1，和改版前一模一样 —— 没有顺带改动别处的断行。

### Notes

- 侧栏的第一项和右边 28px 标题**字心齐平**（都在 y≈33.5）：
  `top: 0` + `padding-top: 14px`。`scrollY = 0` 时 sticky 也把这个值当约束，
  所以上下都不会跳。
- 指示条画在项**内部**（`left: 0`）：`.section-tab` 带 `overflow: hidden`
  （水波纹要裁），伸出盒子外会被切掉。项高 ≈39px、圆角 8px，左侧直边只剩 y=8..31，
  18px 高居中正好落在 10.5..28.5，啃不到圆角。
- AI 抽屉仍然是 `position: fixed`、**不是 grid item**，所以 1280px 两栏 /
  640px 单列两个断点一个都没动。
- **`--series-*` 一个值都没碰**（仍过 `validate_palette.js`）；
  新样式只用 `--md-*` / `--page` / `--text-*`。

### Tests

- 四个宽度逐项 DOM 度量确认（1440 rail / 1280 rail / 800 横条 / 375 横条）：
  grid 分列、`--nav-h` 0 与非 0、`scroll-margin-top`、rail 粘顶位置、
  跳转公式复算（跳到「标度验证」后 `getBoundingClientRect().top === stickyTop + 12` 精确成立）、
  底部钉住高亮、`scrollWidth == clientWidth` 无横向溢出、7 项统计量仍排一行。
- 深浅色各读一遍：指示条 / 浅底 / 分界线 / `--series-1` 分别是
  `#0b0b0b`·`#e4e3db`·`#2a78d6`（浅）与 `#ffffff`·`#2c2c2a`·`#3987e5`（深）——
  系列色与改版前一致。
- 控制台无报错，无失败请求；`node --check` 四个 JS 文件全过。

---

## [2026-09-22] 前端改版：Material 3、分区导航、状态与导出

这一节是页面这一侧的改动，后端接口在下面那一节里。

### Added

- **力–伸长曲线**（「标度验证」分区第一张卡）。横轴是无量纲力
  `x = f·l/(k_BT)`，纵轴是伸长比 `λ = ⟨x⟩/(n·l)`：实线是 Langevin 反函数，
  虚线是高斯链的 `λ = x/3`（只画到 x = 3），另有一条竖线标出两条线**相差 2%** 的位置。
  填 λ 可读出 x、力（pN）、绝对伸长；鼠标悬停有十字准线和浮窗（x / λ / f / ⟨x⟩）。
  曲线本身与 n、l **无关** —— Langevin 函数里没有它们，n、l 只在换算绝对伸长和 pN 时进来。
  所有数值都是从 `/api/force` 返回的 400 点采样里**读**出来的，页面上不写公式。
- **h → n 反解**。填目标末端距、选它指的是哪一种 h（h_rms / ⟨h⟩ / h\* / 全伸展 nl），
  给出连续解 `n_exact` 以及 ⌊n⌋、⌈n⌉ 两侧整数解各自的 h，并可一键「用 ⌈n⌉ 作为 n」。
  说明里点明 `n ∝ (h/l)²` —— 方向和统计量那边正好相反，l 翻倍只要 1/4 的链段数。
- **Kuhn 长度预置**（真实高分子）。8 个 chip（DNA、PE、PS、PMMA、PDMS、顺式聚异戊二烯、
  PEO、蛋白质主链），点一个就把顶部的 `l` 填成它的 Kuhn 长度 `b`，顺带显示持久长度
  `p = b/2`、可信度标签和出处说明。数据**只有 `fjc_core.KUHN_PRESETS` 一份**，
  页面从 `GET /api/kuhn` 拉，不在 JS 里抄第二份 —— 抄了就会改一处忘一处。
  可信度标签是必须的：同一种聚合物的 b 在文献里能差一倍。
- **顶部 sticky App Bar + 分区标签（scrollspy）**。四个分区：参数与结果 / 可视化 /
  标度验证 / 数值表。滚动时自动高亮当前分区，点击平滑跳到该分区；
  栏高量出来写进 `--bar-h`，`scroll-margin-top` 跟着它走，标题不会被吸顶栏盖住；
  滚到文档底部时把最后一个标签钉成高亮。栏本身用负外边距撑满内容区宽度，
  所以滚动时正文不会从栏底下漏出来。
- **URL 状态**：`?n=…&l=…&norm=1` 写进地址栏（`history.replaceState`），
  刷新或把链接发给别人时**以 URL 为准** —— 否则对方打开看到的是他自己存的旧参数。
  优先级：URL > localStorage > 默认值。
- **记住上次参数 + 一键重置**。参数存 localStorage；「恢复默认」把 n / l / 归一化
  连同 Kuhn 读数、反解读数一起清掉。只有**计算成功**才写状态 —— 打错字的中间值
  不会覆盖掉好状态。
- **三张 SVG 图都能导出 PNG**：P(h)、力–伸长、链长扫描。共用 `app.js` 里一份
  `svgToPng`：把每个元素的**计算样式**逐个内联进克隆树、垫一张卡片底色、
  再按 2× 光栅化。序列化出来的 SVG 是独立文件，看不见 `style.css` 也读不到
  `var(--series-*)`，不内联就一定丢颜色。三张图各写一份的话，导出的样子迟早不一样。
- **M3 水波纹**：按钮 / chip / 标签按下时在指针位置扩散一圈，`prefers-reduced-motion`
  下自动关掉。用事件委托实现，因为 chip 是动态生成的，逐个绑会漏。

### Changed

- **整套界面换成 Material 3**，手写 CSS，**不引 CDN、不引框架**，离线照旧可用：
  - 新增一组 `--md-*` 界面角色 token，和数据用的 `--series-*` **彻底分开**；
  - 按钮分 Filled（主操作）/ Tonal（次级）/ Outlined（导出、清空）三态；
  - 填充式文本框：底色填充 + 只留一条下边线，focus 时下边线 1→2px 并用
    `padding-bottom` 减 1px 抵掉，否则一聚焦整个表单往下跳一行；
  - chip 改成 8px 方角（原来是药丸形）；卡片 12dp 圆角；tonal 表面四级分层；
  - M3 字阶：display 57 / headline 24 / title 16 / body 14 / label 12。
- **主色用中性墨色，不用色相**：浅色下 `--md-primary` 是 `#0b0b0b`、深色下是 `#ffffff`。
  理由是本项目「颜色只用来表示数据系列」，而且深色下白字压在 `--series-1` 上
  只有 3.6:1 对比度 —— 这条 `style.css` 里原本就记着，M3 允许中性配色，
  形状 / 阴影 / 状态层 / 字阶足以撑起「谷歌风」。
  **`--series-1..5` 在三个主题下的值一个都没动**（那套色板是跑过
  `validate_palette.js` 校验的）。
- 页面副标题从吸顶栏挪到「参数与结果」的导语里：两行副标题会吃掉 768px 高视口的
  约七分之一，吸顶栏只留标题 + 分区标签 + 两个按钮。
- 内容重新分组：**参数与结果**（含 Kuhn、反解、主结果、统计量卡片）/
  **可视化**（P(h) ‖ 3D，保持同屏配对）/ **标度验证**（力–伸长 + 链长扫描，
  两张图都是在验证什么，所以放一节）/ **数值表**。
- 中文行高保留 1.65，没有照抄 M3 的 20px —— M3 的字阶是按拉丁文调的。

### Fixed

- 反解说明里的 `⌊n⌋` / `⌈n⌉` 丢了字母 n，显示成 `⌊⌋` / `⌈⌉`。
- `svgToPng` 原来不返回值、光栅化失败时只 `revokeObjectURL` 不吭声。
  现在返回 Promise（带 blob），失败在对应卡片的错误条里写「导出 PNG 失败：…」。
- 窄栏里按钮会断成两行（「回到 l = 1」变成「回到 l =」「1」）→ 按钮统一
  `white-space: nowrap`。
- n 的 7 个预设 chip 在 330px 下会把 `10000` 挤到单独一行 → `.chips` 上限放宽到 400px
  （实测 7 个共 380px；368 还是差一点）。

### Tests

- 后端 `python -m unittest test_fjc_core test_ai` → **278 项全过**
  （`test_fjc_core` 127 项 + `test_ai` 151 项）。
- 端到端过了一遍：`/api/kuhn`、`/api/force` 均 200、控制台无报错；
  scrollspy 高亮与跳转、Kuhn chip 回填 `l`、反解读数与「用 n = …」、
  λ 读数、力–伸长悬停浮窗、深浅色、375px 窄屏、三张图的 PNG 导出、
  URL ↔ localStorage ↔ 重置的往返 —— 逐项确认。

---

## [2026-09-22] 后端：力–伸长、h→n 反解、Kuhn 预置表

### Added

- `fjc_core.inverse_langevin()`（二分法求 Langevin 反函数）与
  `fjc_core.force_extension()` → **`POST /api/force`**，返回 400 点的
  `x / λ / 绝对伸长 / 力(pN)`，外加高斯对照线、λ 到达 `x_max` 的位置、
  以及两条线开始相差 2% 的 x。
- `fjc_core.solve_n()` → **`POST /api/solve_n`**，四种 h 各有解析式，
  返回连续解与两侧整数解（以及各自算回去的 h）。
- `fjc_core.KUHN_PRESETS`（8 条）+ **`GET /api/kuhn`**，附 `conf_label` 可信度标签：
  DNA 是硬值、PE 是由定义式推出、其余标「文献标称值」。
- `ai_tools.py` 新增 `solve_n`、`force_extension` 两个服务端工具；
  `ai_local.py` 新增「反解」「力–伸长」两个意图（含「拉到 60% 需要多大力」
  「h=37 要多少个链段」两种问法）。`ai_tools.TOOLS` 现在 **16 个**
  （服务端 7 + 前端 9）。
- 新增测试：反解与内核逐字段对拍、算出来再反解得回原来的 n、
  `force_extension` 不返回整条曲线（体积护栏）、参数名是 `lam` 不是 `lambda`
  （`lambda` 在 Python 里不能当关键字参数）、λ 越界要报错、
  以及「只有 n 会改变绝对伸长」这类不变式。合计 278 项。

### Notes

- 单位换算 `f[pN] = x·0.01380649·T/l_nm` 与它的精确线性逆都在后端 ——
  前端拿到什么读什么，不复算。
- 相关常量：`CURVE_POINTS = FORCE_CURVE_POINTS = 400`、`FORCE_X_MAX = 100`、
  `DEFAULT_TEMPERATURE = 298.15`。

---

## [2026-09-22] AI 助手接入设置（endpoint / model / key 全填）

### Added

- 抽屉里加「接入设置」表单：**endpoint / model / key 三栏都能改**，
  保存走 `POST /api/ai/config` 写回 `ai_config.json`，**下次提问即生效，不用重启**；
  另有 `POST /api/ai/test` 测试连接、服务商预设下拉框。
- 密钥栏是 `type="password"` 且**从不回填**：页面上只有掩码，留空的语义是「不改」，
  要清得显式点「清除密钥」（`clear_key`）—— 否则用户一进表单点保存就会把真 key
  覆盖成 `••••3333`。

### Changed

- 推翻了原先「密钥不进浏览器」的决定。代价是**本机任意能访问 `127.0.0.1:8770`
  的进程**都能改掉 key 或把 endpoint 指到别处；换来的是一次配好、页面上直接改。
  仍然守住的：响应体里**永远没有 key 原文**（含测试连接失败时网关回显 key 的错误，
  必须先 scrub）、`/api/ai/*` 只收 `application/json`、不加任何 CORS 头、
  **助手（模型）不能改网关配置** —— 配置表单不在 `ai_tools.TOOLS` 里，
  也不在 `ACTIONS` 里；`test_ai.py` 有一条断言专门盯着「配置表单进不了动作表」。
  这条反过来的理由写在 `server.py` 头部：能改 endpoint 就等于能把下一问
  连同密钥发到攻击者挑的地方，所以**只有用户点按钮**能走到那里。

---

## [2026-09-22] AI 助手

### Added

- `ai_tools.py`（工具 schema + 服务端 handler，直接调 `fjc_core`，**不碰网络**）、
  `ai.py`（配置 + stdlib `urllib` 传输 + agent 循环 + 会话 + 本地兜底）、
  `ai_local.py`（本地模式的关键词解析）。
- 五个路由：`GET /api/ai/status`、`POST /api/ai/config`、`POST /api/ai/test`、
  `POST /api/ai/ask`、`POST /api/ai/resume`。
- `web/ai.js`：右侧 380px 固定抽屉（不进两栏布局，1280px / 640px 两个断点一个都没动）。
- **本地模式**：没配 key 或网关挂了时按关键词调同一批工具，答案全部来自真实计算，
  认不出就直说认不出，**绝不猜数字**。
- 函数调用降级链：带 `tools` → 没有 `tool_calls` 就从正文里抠围栏 JSON → 都没有就当最终回答。
  系统提示词里**也写了一份**工具说明，所以第二条路真能走通。
- 前端工具走 `window.fjcApp` / `window.fjc3d` / `window.fjcSweep` 三个显式接口，
  每个 setter 都走界面自己的那条路（派发事件或调它自己的刷新），图才会真的重画。

### Security

- 正文一律 `textContent` 渲染，不解析 Markdown —— 模型输出和用户输入都是不可信内容。
- 导出类动作（PNG / CSV / 完整 JSON / 复制）**不是**助手的工具，仍由用户点按钮。
- 对话历史只在内存里，刷新即清，不写 `data/`。
- 硬上限：6 轮工具调用、单次 HTTP 20 秒、整体 60 秒墙钟，三者取「与」。

---

## [2026-09-22] 3D 单链构象

### Added

- `web/chain3d.js`：**手写的 canvas 2D 正交投影**，没有引入 Three.js ——
  保住「无框架、无 CDN、离线可用」。拖动旋转、滚轮缩放、双击复位、
  聚焦后方向键旋转，支持多条链与生长动画。
- canvas 不参与 CSS 变量解析，所以每次重绘都要
  `getComputedStyle(root).getPropertyValue('--series-1')` 现取，
  否则切换深浅色后画布颜色不会跟着变。

---

## [2026-09-22] 链长扫描

### Added

- `fjc_core.sweep()` + **`POST /api/sweep`**：双对数坐标下验 `⟨R²⟩ = n·l²`
  是不是一条斜率恰为 1 的直线。
- `web/sweep.js`：扫描卡片（参数、运行、悬停读数、汇总行、CSV / JSON 导出）。
- 计算量预算是**表现层约束，不是物理约束**：n ≤ 20、链数 ≤ 100000、
  Σ(链数×n) ≤ 60M；都对应 `validate_sweep()` 的可关参数。
- 每个 n 用 `SeedSequence(seed).spawn(k)` 派生子种子，
  **第 i 个位置的随机流只由 i 决定**，往序列里加一个 n 不会打乱其余的点。

---

## [2026-09-22] 首版

### Added

- `fjc_core.py`：统计量（⟨h²⟩、h_rms、h\*、⟨h⟩、σ、Rg、nl、Cn）、
  400 点的径向分布 P(h)、单链随机模拟、自由旋转链交叉验证。**不 import flask，可单独 import。**
- `server.py`：`/`、`POST /api/compute`、`POST /api/chain`，全部**无状态**，
  `FJCInputError` 转 400 加中文信息。
- `web/index.html` + `app.js`：输入 n 得 h、多重对比（最多 5 个 n）、归一化显示、
  P(h) 曲线与特征线、统计量卡片、制表符分隔的数值表（可一键复制进 Excel）。
- 深浅色两套 token；SVG 的 fill/stroke 认 CSS 变量，切深浅色图自动跟着变。
- `启动.cmd`：找 anaconda、找不到就退回裸 `python`。**纯 ASCII + CRLF**（见 README）。
