**## 1. 現在のディレクトリ構造**

```text

filfinder_filament/

├── filfinder_simajiriteacher.py             # strict側：物理スケールで抽出・長さ計測・FITS/CSV/PNG保存

├── filfinder_simajiriteacher_loose.py       # loose側：別条件で抽出し、専用出力先に保存

├── filfinder_scripts_guide.md             # 本ガイド

├── archive_old_experiments/

│   ├── filfinder_2class/

│   │   ├── filfinder_test_split_imgs.py    # 左右・上下の座標で分割し、膨張した0/1ラベルを保存

│   │   └── dilation_colored.py             # 骨格の3×3膨張前後を着色して比較

│   ├── filfinder_test_save/

│   │   ├── filfinder_test_save_imgs.py     # IC5146の入力＋骨格・マスクの確認画像

│   │   ├── filfinder_test_save_imgs_ic5146.py # IC5146の背景除外・検出・256pxパッチ生成

│   │   ├── filfinder_test_save_imgs_orionB.py # 実際はtauN3：検出からパッチ保存まで一体処理

│   │   ├── filfinder_test_save_imgs_tarus.py # 実際はIC5146：物理柱密度閾値・短枝除去・パッチ生成

│   │   └── filfinder_test_save_only.py     # Orion B：再検出して可視化だけを保存

│   ├── filfinder_test_save_fix/

│   │   ├── filfinder_test_save_imgs_ic5146_fix.py # 実際はAquila：物理柱密度閾値・16px長さ制限

│   │   └── filfinder_test_save_imgs_aquila.py # Aquila：距離に応じたpc単位の閾値を試行

│   ├── old_dataset_split/

│   │   └── filfinder_test_train_val.py     # 768px空間ブロックによるtrain/val分割

│   └── old_prediction_visualization/

│       ├── filfinder_test_save_rgb.py     # 既存PNGの再着色。検出は再実行しない

│       └── filfinder_label_comparision.py # MMSeg予測とFilFinderラベルの一致度評価

├── docs/                                  # 上記の一部と同名のスクリプト・ライブラリ文書

├── fil_finder/                            # FilFinder本体・テスト

├── examples/                              # ライブラリの使用例・論文図用コード

└── images/                                # 画像資料

```

**## 2. 各スクリプトの処理と出力**

以下のファイル名は、第1節の配置に対応する。

| スクリプト | 現コードの対象・設定 | 処理と出力 |

|---|---|---|

| `filfinder_simajiriteacher.py` | Aquila、440 pc、柱密度閾値7.52e21 | 0以下をNaN化→平坦化を省略→マスク→骨格→枝刈り。長さ・robust判定をCSV、柱密度・骨格をFITS、解析オブジェクトをPKL、最長経路をPNG保存。学習パッチ生成はしない |

| `filfinder_simajiriteacher_loose.py` | Polaris、140 pc、柱密度閾値0.94e21 | 同型の抽出・計測をloose用フォルダに保存。最長経路に加え骨格・マスクの重ね合わせを保存。長さのrobust判定は0.1 pc |

| `make_split.py` | Aquila / IC5146、train比0.8、seed42 | 入力と0/1ラベルをペアにして領域ごとに無作為分割。train/valのシンボリックリンクとMMSeg用TXTを生成 |

| `filfinder_test_split_imgs.py` | pipeCenterB59-250、256px、stride64、dataset_name="250115" | crop関数は左右にタイルを保存する設計。実行部は既存right/leftパッチを読み、rightをy座標でtrain/val、leftをtestへ振り分ける。ラベルを3×3膨張し、0/1のパレット画像を生成 |

| `dilation_colored.py` | 既存の骨格PNG | 3×3、1回の膨張前後を白背景に着色して保存。`fits_name` がファイル内で未定義のため、そのまま単独実行はできない。OpenCVのBGR配列に[255,0,0]を指定しており、赤というコメントに反して保存色は青 |

| `filfinder_test_save_imgs.py` | IC5146、436 pc、glob_thresh=100 | 負値をNaN化し、マスクと骨格の3種の確認PNGをmask_testに保存する設計。ファイルにMarkdownコードフェンスが含まれ、そのままでは構文エラー |

| `filfinder_test_save_imgs_ic5146.py` | IC5146、436 pc、glob_thresh=100、stride16 | 下位2%の背景候補を膨張して無効化。flatten_percent=95で検出し、キャッシュ、全面入力・骨格、256px入力/GTパッチ、可視化を保存 |

| `filfinder_test_save_imgs_orionB.py` | 名前と異なりtauN3、436 pc、glob_thresh=100、stride64 | FITS→前処理→マスク→骨格解析→RHTを段階別にキャッシュ。入力と0/1骨格のパッチをPNG/TIFF/PKL、全面・重ね合わせ画像を保存 |

| `filfinder_test_save_imgs_tarus.py` | 名前と異なりIC5146、460 pc、1.88e21、stride16 | 物理単位付き柱密度閾値で検出。枝とフィラメントの長さに16px閾値を指定し、入力/GTパッチ・全面画像・可視化を保存 |

| `filfinder_test_save_only.py` | Orion B、436 pc、1.88e21 | キャッシュを使わず再検出し、マスク・骨格・最長経路をPNG保存。入力/GTパッチは作らない |

| `filfinder_test_save_imgs_ic5146_fix.py` | 名前と異なりAquila、460 pc、7.52e21、stride16 | 物理柱密度閾値、16px枝刈り・長さフィルタを指定。ML用対数正規化と比較用線形正規化を分け、入力/GTパッチと可視化を保存。IC5146指定時のみ負値をNaN化する条件がある |

| `filfinder_test_save_imgs_aquila.py` | Aquila、260 pc（コードに要確認）、7.52e21、stride16 | 距離辞書とWCSを用い、平滑化・適応閾値・枝・最小長さを0.1 pc基準で指定。面積閾値もpc²化し、画素換算値を表示。パッチと可視化を保存 |

| `filfinder_test_train_val.py` | Aquila / IC5146、ブロック768px、パッチ256px | GT値1を含むパッチだけを採用。ブロックをまたぐパッチを除き、ブロック単位で約8:2分割。train/val・対象TXTを保存。BUFFER_SIZE=256は定義のみで未使用 |

| `filfinder_test_save_rgb.py` | 既存Aquila画像（設定で切替） | 保存済みPNGの背景をinferno、既存の赤い検出画素を緑へ再着色し、拡大図を保存。元の検出結果・GTは変えない |

| `filfinder_label_comparision.py` | Aquila / IC5146 / Orion B | 別途用意したMMSeg configとcheckpointで推論し、TP/FP/FN/TNからPrecision・Recall・IoU・F1を集計。metrics_summary.csvとエラー可視化を保存 |

ファイル名より `fits_name` と実行時の設定を優先する。旧ガイドにあった「ic5146.pyはAquila」「ic5146_fix.pyはIC5146」という説明は、現在のarchive内のコードとは一致しないため修正した。

**## 3. 研究方針・出力の違いを時系列で整理**

実験日を裏付けるログや結果CSVは研究スクリプトの配置内では確認できない。したがって下表は、コードの派生関係と処理内容から整理した研究上の段階であり、全ファイルの厳密な作成順・実行順を確定したものではない。日付は明示されたものだけを記載する。

| 段階・日付の根拠 | 研究方針 | 対応するコード | 出力・比較の違い |

|---|---|---|---|

| 1. 初期2クラス実験（dataset_name="250115"。2025-01-15を示す可能性はあるが実行日未確認） | 背景/フィラメントの0/1教師を作り、画像内の座標で分割する | `filfinder_2class/` | 細い骨格を3×3膨張したラベルを保存。rightを上下でtrain/val、leftをtestに使う。膨張前後で正例画素が変わるため、未膨張GTの指標とは直接比較できない |

| 2. 検出からパッチ保存までの一体化（日付未確認） | FilFinderの結果をそのまま疑似教師にして学習データへ変換する | `filfinder_test_save/` のorionB・ic5146・tarus版 | 256pxパッチをstride64または16で出力。領域・背景処理・閾値が版ごとに異なり、同じ枚数・同じGTにはならない |

| 3. 背景・物理閾値・距離の見直し（日付未確認。第2段階内にも修正が含まれる） | 背景由来の検出を抑え、領域ごとの物理スケールを揃える | ic5146版の背景マスク、`filfinder_test_save_fix/` | glob_thresh=100から物理単位付き閾値、固定pxからpc指定への試行。対象天体と距離も変わるため、検出差を閾値だけの効果とは解釈しない |

| 4. 分割の独立性を見直し（日付未確認。第2〜3段階との前後関係未確定） | パッチ単位の無作為分割から空間ブロック分割へ | `make_split.py` → `old_dataset_split/filfinder_test_train_val.py` | stride16/64の重複パッチがtrain/valへ混在する問題への対策。空間版は背景のみのパッチも除くため、分割方法と評価母集団の両方が変わる。精度改善の実測値は未確認 |

| 5. モデルとの一致度と図の確認（日付未確認） | 学習済みモデルとFilFinder疑似GTを比較し、見逃し・過検出を確認 | `old_prediction_visualization/` | 評価版は数値とTP/FP/FN図、RGB版は見た目だけを変更。評価版は元の全タイルを読むため、独立val/test評価とは区別する |

| 6. 物理基準での抽出・計測を独立化（strict冒頭に2026-07-21修正の記載） | パッチ生成と分けて、骨格・長さ・robust件数を保存する | `filfinder_simajiriteacher.py` | 0以下をNaN化し、平坦化を省略、0.2 pcの平滑化・枝等を指定。長さ0.3 pc以上のrobustフラグをCSV化し、骨格FITSを保存。旧2値パッチ作成とは出力の用途が異なる |

| 7. strictからlooseを派生（loose冒頭に派生関係の記載。作成日未確認） | より緩い条件の候補を別出力として残す | `filfinder_simajiriteacher_loose.py` | loose専用cache/full/visに分離し、マスク・骨格の図を追加。ただし現在はstrict=Aquila、loose=Polarisなので、両者を同一天体の条件比較とみなせない |

| 8. 後続の3クラス・階層型モデル（ユーザー提示の研究方針。実装は別リポジトリ/ZIP） | FilFinder/DisPerSEのstrict・looseから3クラスGTを作り、空間分割・split_v2へ進む。独立2値→共有2-head→HRF事前学習＋共有ASPP／CE＋Dice→Class1/Union比較 | DisPerSE側のGT生成・データ分割、および別ZIPの学習評価コード | 本リポジトリの旧0/1骨格GTとはラベル定義が異なる。Class1、Class2、Unionそれぞれの評価対象と255無効画素を明示する。後続モデルの優劣を示す実測結果はここにはない |

strictの2026-07-21という日付は修正コメントの根拠であり、旧スクリプト全体がそれ以前に実行されたことまでは示さない。並行して検討された可能性がある。

**## 4. 現方針での位置づけ**

```text

FITS柱密度マップ

├─ strict抽出・計測：filfinder_simajiriteacher.py

└─ loose抽出・計測：filfinder_simajiriteacher_loose.py

↓ 同一天体の設定を揃えて別々に保存

FilFinder / DisPerSE の抽出結果

↓ DisPerSE側のラベル生成コード（本リポジトリ外）

3クラスGT（0:背景 / 1:Class1 / 2:Class2 / 255:無効）

↓ 空間分割・正規化・入力/GTパッチ化（split_v2）

別ZIPで学習・PR/F1評価・元FITS座標で可視化

↓ 後続の比較

Head1=Class1 / Head2=Class1∪Class2（Union）

```

直下のstrict/looseは抽出と計測の入口であり、これらだけで3クラスGT、split_v2、モデル学習、Union比較まで実行することはできない。archiveの旧コードは、0/1教師・距離/幅・背景処理・空間分割・可視化の検討経緯を比較する資料として保持する。

**## 5. 結果を比較するときの確認先**

| 比較するもの | 確認する出力・条件 |

|---|---|

| 検出件数・長さ | strict/looseのfull系CSV。n_filaments、n_robust、総長、robust総長、最大長を同一天体・同条件で比較 |

| 抽出位置・形状 | skeleton FITS、マスク・骨格・longpath PNG。距離、WCS、画像反転、無効画素を揃える |

| GTの太さ・クラス | 骨格0/1、膨張0/1、3クラスGTを区別。表示用0/255と学習用クラスIDを混同しない |

| 分割の妥当性 | seed、元座標、stride、ブロック、split対象一覧。背景のみの採否と境界をまたぐパッチを確認 |

| モデルの一致度 | metrics_summary.csv、config、checkpoint、評価対象一覧。全タイル評価と独立val/test評価を区別 |

| 見た目の違い | RGB再着色は表示変更として扱い、検出数やF1の改善とはみなさない |

実測CSVやログがないため、本ガイドに数値結果・モデルの勝敗・strictよりlooseの検出数が増えた等の断定は記載していない。既存の第三者クローンや別環境の結果は調査対象外である。

**## 6. 出力先と実行上の注意**

- strict/looseのFITS・CSV・PNG・PKLは対象天体配下のfull/vis/cache系へ保存する。パッチ生成版はin/out、分割版はsplit/train・split/valと一覧TXTを作る。
- パラメータ変更時はキャッシュの版も管理する。既存PKLが読み込まれると、設定変更が検出結果へ反映されない場合がある。
- 分割スクリプトは既存の出力を削除・再生成する処理を含む。同じsplit出力を使う旧ランダム版と空間版は結果を別途保存して比較する。
- `filfinder_test_train_val.py` のBUFFER_SIZEは未使用であり、明示的バッファ帯が実装済みとは扱わない。
- 初期2クラス版のcrop関数は存在するが、現在の実行部は保存済みパッチの振り分けが中心。座標閾値test_sep_y以上もvalへ入る分岐になっており、right側の下部をtestへ振り分ける実装ではない。
- `filfinder_test_save_imgs.py` のコードフェンスと `dilation_colored.py` の未定義fits_nameは実行前に修正が必要。今回の作業ではコード自体は変更していない。
- 評価で一致させる相手はFilFinder由来の疑似GTである。人手の真値に対する検出精度と同義ではない。
- モデル学習本体・checkpoint・評価CSVは今回の研究用コードには含まれない。後続ZIPとの結果比較には、同じ入力・GT・分割・評価定義を記録する。