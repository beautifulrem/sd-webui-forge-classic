<p align="center">
🌍
<a href="./README.md">English</a> |
<strong>日本語</strong> |
</p>

---

<div align="center">

# Repeat Generate for NEO

---

</div>

[Stable Diffusion WebUI Forge - Neo](https://github.com/Haoming02/sd-webui-forge-classic) で、画像の連続自動生成を行うための拡張機能です。（他のWebUIでも動作する可能性はありますが、Forge Neo 以外での動作確認はしていません）

標準機能の「Generate forever」とは異なり、**Webブラウザを最小化しても生成が止まるなどの問題が発生しません**。

本機能は、以下の機能を対象に画像生成を継続的に再キューイングする軽量なForge拡張機能です：
* txt2img
* img2img

Forge/Gradio のキューAPIを利用して生成の完了を検知し、現在のタスクが終了すると自動的に次の生成を開始します。

## 💕 主な機能

* 自動的な連続画像生成
* キューベースの実行（Forgeネイティブ機能の利用）
* Webブラウザを最小化しても実行を継続
* txt2img と img2img の両方をサポート
* ボタンひとつで連続生成の開始（START）と停止（STOP）を切り替え
* 即座に生成を停止する「Interrupt（中断）」ボタン

## 🖥️ ユーザーインターフェース（UI）

* **↻ Start Repeat / ■ Stop Repeat / ⌛ Stopping**: ボタンは以下の3つの状態に遷移します。

  * **↻ Start Repeat**: 連続生成を開始します。<BR>
    ![Start](/images/start.webp)

  * **■ Stop Repeat**: 連続生成を停止します。（現在の画像生成が完了するまで待機します。）<BR>
    ![Stop](/images/stop.webp)

  * **⌛ Stopping**: 現在の画像生成が完了するまで待ちます。（クリックすると連続生成を再開・再有効化できます。）<BR>
    ![Stopping](/images/stopping.webp)

* **Interrupt**: 連続生成を終了し、さらに現在実行中の画像生成も即座に中断します（Interruptシグナルを送信します）。
* **Status**: 現在のステータス（状態）を表示します。

## ⚠️ 既知の制限事項

連続生成は、開始した時点と同じパラメータを使用して実行され続けます。

* ControlNetは有効にしても無視されます。それ以外にも無視される設定があるかもしれません。
* 画像生成が完了した際、プレビューに最初の画像が一時的に表示されます。
* 連続生成の実行中にパラメータを変更しても反映されません。
* パラメータを変更したい場合は、一度連続生成を停止してから再度開始してください。
  * ボタンが「↻ Start Repeat」に戻るまで待つ必要はありません。「⌛ Stopping」の状態のときにクリックして再開できます。

## 🛠️ インストール方法

1. Stable Diffusion WebUI Forge - Neo を起動します。
2. 「Extensions」タブ → 「Install from URL」を開きます。
3. URL欄に以下を貼り付けます：
   `https://github.com/hirorohi03/repeat-generate-4NEO.git`
4. 「Install」ボタンをクリックし、その後 WebUI を再起動（Restart UI）してください。

## 🚗 動作原理

本拡張機能は以下の流れで動作しています：

1. Forge の `submit` 関数をフックする
2. 生成時の引数（パラメータ）をキャプチャする
3. `/queue/join` へ直接リクエストを送信する
4. `requestProgress()` を使用して完了状態を監視する
5. 完了後、自動的に次の生成リクエストを送信する

これにより、タイマーによる定期ポーリング（監視）を回避し、Forge ネイティブのキューシステムと連携して動作します。

## 🔍 内部状態（ステータス）

拡張機能の内部では、以下の状態を個別に管理しています：

| 状態（State） | 意味 |
| :--- | :--- |
| `running` | 連続生成モードが有効 |
| `busy` | 画像生成タスクが現在実行中 |

この管理により、以下のような正確な挙動を実現しています：
* 現在のタスクが終了した後に安全に停止する
* キューへの二重送信（重複サブミット）を防止する
* UI上に適切なステータスを表示する

## ⚖️ ライセンス / License

本リポジトリは、ライセンスが明記されていなかったアップストリームのリポジトリをフォークしたものです。

元のリポジトリの開発者である**akirau-ai氏**（[@akirau-ai](https://github.com/akirau-ai)）は、[note.com](https://note.com/akirau338/n/n3fbfeaa6184d?scrollpos=comment) にて「特定のライセンスは想定しておらず、誰でも自由に利用してよい」旨を表明されています。

本リポジトリにおける私（hirorohi03）の独自の変更・修正部分に関しては、**MIT License** のもとで公開します。
