#ifndef PTXSYNTHESIST_MAIN_WINDOW_H
#define PTXSYNTHESIST_MAIN_WINDOW_H

#include <QMainWindow>

#include <memory>

class QLineEdit;
class CodeEditor;
class QToolBar;
class QTimer;
class SyntaxStyle;

class PTXExecutor;

class MainWindow : public QMainWindow
{
  Q_OBJECT

public:
  MainWindow();
  ~MainWindow();

private:
  QLineEdit *options {};
  CodeEditor *cuda {};
  CodeEditor *ptx {};
  QTimer *timer {};
  QToolBar *tool_bar {};

  SyntaxStyle* syntaxStyle;

  std::unique_ptr<PTXExecutor> executor;

  void load_style(QString path);

private slots:
  void reset_timer();
  void regen_ptx();
  void interpret();
  void execute();
};

#endif //PTXSYNTHESIST_MAIN_WINDOW_H
