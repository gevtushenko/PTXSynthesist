#ifndef PTXSYNTHESIST_MAIN_WINDOW_H
#define PTXSYNTHESIST_MAIN_WINDOW_H

#include <QMainWindow>

class QLineEdit;
class QTextEdit;
class QToolBar;
class QTimer;

class MainWindow : public QMainWindow
{
  Q_OBJECT

public:
  MainWindow();

private:
  QLineEdit *options {};
  QTextEdit *cuda {};
  QTextEdit *ptx {};
  QTimer *timer {};
  QToolBar *tool_bar {};

private slots:
  void reset_timer();
  void regen_ptx();
  void interpret();
};

#endif //PTXSYNTHESIST_MAIN_WINDOW_H
