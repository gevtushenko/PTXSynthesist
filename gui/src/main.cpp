#include <QApplication>
#include <main_window.h>

int main(int argc, char *argv[])
{
  QApplication a(argc, argv);

  MainWindow main_window;
  main_window.show();

  return QApplication::exec();
}
