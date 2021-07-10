#ifndef PTXSYNTHESIST_MAIN_WINDOW_H
#define PTXSYNTHESIST_MAIN_WINDOW_H

#include <QMainWindow>

class QTextEdit;
class QTimer;

class MainWindow : public QMainWindow
{
  Q_OBJECT

public:
  MainWindow();

private:
  QTextEdit *cuda {};
  QTextEdit *ptx {};
  QTimer *timer {};

private slots:
  void reset_timer();
  void regen_ptx();
};

#endif //PTXSYNTHESIST_MAIN_WINDOW_H
