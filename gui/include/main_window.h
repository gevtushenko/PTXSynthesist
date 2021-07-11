#ifndef PTXSYNTHESIST_MAIN_WINDOW_H
#define PTXSYNTHESIST_MAIN_WINDOW_H

#include <QMainWindow>

#include <memory>

class QLineEdit;
class CodeEditor;
class QToolBar;
class QTimer;
class QLineSeries;
class QChartView;
class QChart;

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

  unsigned int execution_id {};
  QLineSeries *series {};
  QChartView *chart_view {};
  QChart *chart {};

  float min_elapsed = std::numeric_limits<float>::max();
  float max_elapsed = 0.0f;

  SyntaxStyle* syntaxStyle;

  QAction *run_action;
  QAction *interpret_action;

  std::unique_ptr<PTXExecutor> executor;

  void load_style(QString path);

private slots:
  void reset_timer();
  void regen_ptx();
  void interpret();
  void execute();
};

#endif //PTXSYNTHESIST_MAIN_WINDOW_H
