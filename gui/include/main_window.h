#ifndef PTXSYNTHESIST_MAIN_WINDOW_H
#define PTXSYNTHESIST_MAIN_WINDOW_H

#include <QMainWindow>

class QLineEdit;
class CodeEditor;
class QToolBar;
class QTimer;
class SyntaxStyle;

class MainWindow : public QMainWindow
{
  Q_OBJECT

public:
  MainWindow();

private:
  QLineEdit *options {};
  CodeEditor *cuda {};
  CodeEditor *ptx {};
  QTimer *timer {};
  QToolBar *tool_bar {};

  SyntaxStyle* syntaxStyle;

  void load_style(QString path);

private slots:
  void reset_timer();
  void regen_ptx();
  void interpret();
};

#endif //PTXSYNTHESIST_MAIN_WINDOW_H
